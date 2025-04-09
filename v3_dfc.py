from typing import Optional, List, Tuple
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] or int,  # 更新类型提示以指
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)  #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        '''if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()'''
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            #self.rbr_skip=None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding,
                                              groups=1))  # 对于常规和 scale 分支使用默认 groups=1
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            '''if self.kernel_size != (1, 1):  # 如果kernel_size不是(1,1)，即存在实质性的空间卷积
                self.rbr_scale = self._conv_bn(kernel_size=1,  # 使用元组的第一个元素作为kernel_size
                                               padding=0,
                                               groups=1)'''  # 此处省略的代码不变'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))
            #return self.reparam_conv(x)

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(out)
        #return out


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if isinstance(self.kernel_size, tuple):
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        else:
            pad_h = pad_w = (self.kernel_size - 1) // 2
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size
            if isinstance(self.kernel_size, tuple):
                pad_h = (self.kernel_size[0] - 1) // 2
                pad_w = (self.kernel_size[1] - 1) // 2
                padding = [pad_w, pad_w, pad_h, pad_h]
            else:
                padding = [pad_h, pad_h, pad_w, pad_w]
            kernel_scale = torch.nn.functional.pad(kernel_scale, padding)
        # ...

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            *self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int,
                 groups: int = 1) -> nn.Sequential:  # 添加 groups 参数，且默认值为 1
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :param groups: Number of groups for the convolution.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,  # 使用此处的 groups 参数
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list



class MobileOne(nn.Module):
    """ Simplified MobileOne Model with only one stage containing 3 branches of
    Depthwise and Pointwise convolutions at training time.
    """

    def __init__(self, inp: int, oup: int,dw_size=5,
                  width_multiplier: float = 1.0,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 num_conv_branches: int = 3, inference_mode: bool = False,
                 deploy=False, ratio=2,mode=None):
        super().__init__()
        self.blocks = nn.ModuleList()  # 使用ModuleList来存储所有的块
        self.deploy = deploy
        self.mode = mode
        self.gate_fn = nn.Sigmoid()
        self.scale = 1.0
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.num_conv_branches = num_conv_branches
        in_planes = 3  # 输入通道数，例如，彩色图像为3
        out_planes = int(32 * width_multiplier)  # 根据需求调整通道数，并应用宽度乘数
          # 假设每个MobileOneBlock有3个分支
        # Pointwise Conv
        if self.mode in ['original']:
            self.primary_conv = MobileOneBlock(in_channels=inp, out_channels=init_channels, kernel_size=(1, 1),
                                               stride=(2, 1), padding=0,groups=1, inference_mode=inference_mode,
                                               use_se=None,num_conv_branches=3)

            self.cheap_operation =MobileOneBlock(in_channels=init_channels, out_channels=new_channels, kernel_size=(dw_size, 1),
                                               stride=(1, 1), padding=((dw_size-1)//2,0), groups=init_channels, inference_mode=inference_mode,
                                               use_se=None,num_conv_branches=3)

        elif self.mode in ['attn']:
            self.primary_conv = MobileOneBlock(in_channels=inp, out_channels=init_channels, kernel_size=(1, 1),
                                               stride=(2, 1), padding=0, groups=1, inference_mode=inference_mode,
                                               use_se=None, num_conv_branches=3)

            self.cheap_operation = MobileOneBlock(in_channels=init_channels, out_channels=new_channels,
                                                  kernel_size=(dw_size, 1),
                                                  stride=(1, 1), padding=((dw_size - 1) // 2, 0), groups=init_channels,
                                                  inference_mode=inference_mode,
                                                  use_se=None, num_conv_branches=3)
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=(2,1), padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,1), stride=1, padding=(0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2,0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )



    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)

            return out
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=(2,1), stride=(2,1)))
            # res = self.short_conv(F.max_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            # print('x1',x1.shape)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)

            return out * F.interpolate(self.gate_fn(res), size=(x2.shape[-2], x2.shape[-1]),
                                                 mode='nearest')

    def reparameterize(self):
        """ 在支持的所有阶段上调用重参数化。 """
        if hasattr(self.primary_conv, 'reparameterize'):
            self.primary_conv.reparameterize()
        if hasattr(self.cheap_operation, 'reparameterize'):
            self.cheap_operation.reparameterize()

    def switch_to_deploy(self):
        self.reparameterize()



class RepGhostv2(nn.Module):
    """RepGhost bottleneck w/ optional SE and support for 'original' and 'attn' modes based on layer_id"""

    def __init__(
            self,
            classifier=12,
            dw_kernel_size=3,

            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
            layer_id=0,
            args=None
    ):
        super(RepGhostv2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0

        self.enable_shortcut = shortcut


        # Determine mode based on the layer_id as in GhostBottleneckV2


        # Point-wise expansion
        self.ghost1 = MobileOne(
            1,
            64,
            stride=(2,1),
            #relu=True,
            mode='attn',
            deploy=deploy,
        )

        self.ghost2 = MobileOne(
            64,
            128,
            #dw_size=5,
            stride=(2, 1),

            mode='attn',  # Always 'original' as in GhostBottleneckV2

            deploy=deploy,
        )


        self.ghost3 = MobileOne(
            128,
            256,
            # dw_size=5,
            stride=(2, 1),

            mode='attn',  # Always 'original' as in GhostBottleneckV2

            deploy=deploy,

        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, 6))
        self.classifier = nn.Linear(256 * 6, classifier)


    def forward(self, x):
        # 1st repghost bottleneck
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.ghost3(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    '''def forward(self, x):

        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.ghost3(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x'''


def repghost_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    # 如果 do_copy 设置为 True，则转换模型到部署模式的函数将首先复制一份模型，
    # 然后在副本上进行所有修改，保留原始模型不变。这样做的好处是你还可以保有原始模型，以便需要时使用或重新训练。
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model, save_path)
    return model


if __name__ == '__main__':
    model = RepGhostv2().eval()
    print(model)
    # from tools import cal_flops_params

    # flops, params = cal_flops_params(model, input_size=(1, 1, 128, 9))
    input = torch.randn(1, 1,171,40)
    # 确保模型处于评估模式
    model.eval()
    from tools import cal_flops_params

    flops, params = cal_flops_params(model, input_size=input.shape)

    # 将模型转换为部署模式
    deployed_model = repghost_model_convert(model, do_copy=False)

    # 打印部署后的模型结构
    print(deployed_model)

    # 如果已经在 get_equivalent_kernel_bias 方法中添加了打印语句，则在调用上面的函数时应该打印出 kernel3x3 的尺寸
    import sys

    sys.path.append("../")
    from tools import cal_flops_params

    flops, params = cal_flops_params(model, input_size=input.shape)
