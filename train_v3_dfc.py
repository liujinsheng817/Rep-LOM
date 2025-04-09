import json
import os
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.autograd import Variable
from torchvision import datasets
from model.v3_dfc import RepGhostv2
import numpy as np

torch.backends.cudnn.benchmark = False
import warnings

warnings.filterwarnings("ignore")
'''torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False'''

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch, model_ema):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print('训练集总数', total_num, '加载次数', len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device, non_blocking=True), Variable(target).to(device,
                                                                               non_blocking=True)
        #samples, targets = mixup_fn(data, target)
        samples, targets = data, target

        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = nan_to_num(criterion_train(output, targets))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            loss = criterion_train(output, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 80 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc


def nan_to_num(tensor, nan=0.0, posinf=None, neginf=None):
    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, nan), tensor)
    if posinf is not None:
        tensor = torch.where(torch.isinf(tensor) & (tensor > 0), torch.full_like(tensor, posinf), tensor)
    if neginf is not None:
        tensor = torch.where(torch.isinf(tensor) & (tensor < 0), torch.full_like(tensor, neginf), tensor)
    return tensor


# 验证过程
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []
    for data, target in test_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    print('标签大小', target.size(0))
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))

    # 在每次验证过程后，添加以下代码来记录模型
    if acc > Best_ACC:
        Best_ACC = acc
        # 保存最佳模型的状态，包括周期
        '''best_model_info = (acc, f'model3.5_{epoch}_{round(acc, 3)}_best.pth')
        # 更新epoch_acc_dict
        epoch_acc_dict[epoch] = best_model_info'''

    '''if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), file_dir + '/' + 'best.pth')
        else:
            torch.save(model.state_dict(), file_dir + '/' + 'best.pth')
        Best_ACC = acc'''
    '''if isinstance(model, torch.nn.DataParallel):
        state = {

            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'Best_ACC': Best_ACC
        }
        torch.save(state, file_dir + "/" + 'model3.5_' + str(epoch) + 'X735_' + str(round(acc, 3)) + '.pth')
    else:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'Best_ACC': Best_ACC
        }
        torch.save(state, file_dir + "/" + 'model3.5_' + str(epoch) + 'X735_' + str(round(acc, 3)) + '.pth')'''
    return val_list, pred_list, loss_meter.avg, acc


if __name__ == '__main__':
    # 创建保存模型的文件夹
    file_dir = 'checkpoints/RepGhostv3/usc'
    if os.path.exists(file_dir):
        print('true')

        os.makedirs(file_dir, exist_ok=True)
    else:
        os.makedirs(file_dir)

    # 设置全局参数
    model_lr = 1e-3  # 1e-4
    BATCH_SIZE = 128
    EPOCHS = 150
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # 是否使用混合精度
    use_dp = False  # 是否开启dp方式的多卡训练
    classes = 12
    resume = None  # 是否接着上次模型继续训练，如果不为空，则按照resume的值加载模型。如果是None则表示不接着上次训练的模型训
    CLIP_GRAD = 5.0  # 梯度的最大范数，在梯度裁剪里设置
    Best_ACC = 0  # 记录最高得分
    use_ema = False
    model_ema_decay = 0.9998  # ema衰减值
    start_epoch = 1
    # 数据预处理7
    '''transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])
    ])'''
    '''mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=classes)'''


    # 读取数据

    def get_ImageNet_train_dataset():
        # 使用新的数据集加载方式
        train_feature = np.load('sport data/Pampa2/x_train.npy')
        train_label = np.load('sport data/Pampa2/y_train.npy')

        train_feature_tensor = torch.from_numpy(train_feature).float().unsqueeze(1)
        train_label_tensor = torch.from_numpy(train_label).long()

        train_dataset = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)

        return train_dataset


    def get_ImageNet_val_dataset():
        # 使用新的数据集加载方式
        test_feature = np.load('sport data/Pampa2/x_valid.npy')
        test_label = np.load('sport data/Pampa2/y_valid.npy')

        test_feature_tensor = torch.from_numpy(test_feature).float().unsqueeze(1)
        test_label_tensor = torch.from_numpy(test_label).long()

        val_dataset = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)

        return val_dataset


    dataset_train = get_ImageNet_train_dataset()
    dataset_test = get_ImageNet_val_dataset()

    '''with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))'''
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化模型并且移动到GPU
    criterion_train = torch.nn.CrossEntropyLoss()
    criterion_val = torch.nn.CrossEntropyLoss()
    # 设置模型
    model_ft = RepGhostv2()
    #print('网络结构', model_ft)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, classes)
    if resume:
        model = torch.load(resume)
        print(model['state_dict'].keys())
        model_ft.load_state_dict(model['state_dict'])

        Best_ACC= model['Best_ACC']
        start_epoch = model['epoch'] + 1
    model_ft.to(DEVICE)
    print(model_ft)
    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr)
    #cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # T_0就是初始restart的epoch数目
        T_mult=1,  # T_mult就是增加T_0的乘法因子，即经过一次restart T_0 = T_0 * T_mult
        eta_min=5e-7,  # 最小学习率
    )
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)
    if use_ema:
        model_ema = ModelEma(
            model_ft,
            decay=model_ema_decay,
            device=DEVICE,
            resume=resume)
    else:
        model_ema = None

    # 训练与验证
    is_set_lr = False
    log_dir = {}
    best_model_state = None
    epoch_acc_dict = {}
    best_acc = 0
    best_models = []
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch, model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        if use_ema:
            val_list, pred_list, val_loss, val_acc = val(model_ema.ema, DEVICE, test_loader)
        else:
            val_list, pred_list, val_loss, val_acc = val(model_ft, DEVICE, test_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                # 提取当前最佳模型状态
                '''best_model_state = {
                    'epoch': epoch,
                    'state_dict': model_ft.module.state_dict() if isinstance(model_ft,
                                                                             torch.nn.DataParallel) else model_ft.state_dict(),
                    'Best_ACC': best_acc
                }'''
        current_model_state = {
            'epoch': epoch,
            'state_dict': copy.deepcopy(
                model_ft.module.state_dict() if isinstance(model_ft, torch.nn.DataParallel) else model_ft.state_dict()),
            'val_acc': val_acc
        }

        if val_acc >= 70:
            best_models.sort(key=lambda x: x['val_acc'], reverse=True)
            if len(best_models) < 50 or val_acc > best_models[-1]['val_acc']:
                best_models.append(current_model_state)
                # 再次排序以确保列表是按准确率降序排列的
                best_models.sort(key=lambda x: x['val_acc'], reverse=True)
                # 保持列表总是有15个最佳模型
                best_models = best_models[:50]
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        '''with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))'''
        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)

        # 在所有训练周期后的代码部分
    # 在所有训练周期后的代码部分

    # 在所有训练周期后的代码部分
    best_model_dir = "checkpoints/RepGhostv3/pampa/v3_dfc_5x1_1x1best.3.3"
    os.makedirs(best_model_dir, exist_ok=True)

    for idx, best_model_state in enumerate(best_models):
        best_epoch = best_model_state['epoch']
        best_accuracy = best_model_state['val_acc']
        filename = f'model3.5_{best_epoch}x735_{round(best_accuracy, 5)}_rank{idx + 1}.pth'
        best_model_save_path = os.path.join(best_model_dir, filename)
        torch.save(best_model_state, best_model_save_path)
        print(
            f"Ranked {idx + 1} model at epoch {best_epoch} with val accuracy {round(best_accuracy, 4)} saved to {best_model_save_path}")
