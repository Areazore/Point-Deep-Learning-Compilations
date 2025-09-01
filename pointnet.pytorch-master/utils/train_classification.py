from __future__ import print_function

# =============================================
# PointNet 分类训练脚本（ShapeNet / ModelNet40）
#
# 主要流程：
# 1) 解析命令行参数
# 2) 构建训练/测试数据集与 DataLoader
# 3) 初始化 PointNet 分类模型与优化器/调度器
# 4) 迭代训练，并周期性在测试集上做快速评估
# 5) 训练结束后在完整测试集上计算最终精度
#
# 重要说明（Windows 用户）：
# - 已放入 if __name__ == "__main__": 块以支持多进程 DataLoader。
# - 若多进程仍报错，可将 --workers 改为 0。
# =============================================
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset, ModelNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')  # 输入批次大小
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')  # 每个点云采样点数（PointNet 输入点数）
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)  # DataLoader 进程数
    parser.add_argument(
        '--nepoch', type=int, default=5, help='number of epochs to train for')  # 训练轮数
    parser.add_argument('--outf', type=str, default='cls', help='output folder')  # 模型保存目录
    parser.add_argument('--model', type=str, default='', help='model path')  # 预训练模型路径（可选）
    parser.add_argument('--dataset', type=str, help="dataset path",
                        default="D:\Code\Python\pointnet.pytorch-master\shapenetcore_partanno_segmentation_benchmark_v0")  # 数据集根目录
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")  # 选择数据集类型
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")  # 是否使用特征变换正则

    opt = parser.parse_args()
    print(opt)

    # 随机种子设置（保证可复现）
    blue = lambda x: '\033[94m' + x + '\033[0m'  # 控制台蓝色输出
    opt.manualSeed = random.randint(1, 10000)  # 生成随机种子
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # -----------------------
    # 数据集加载
    # -----------------------
    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)  # 指定分类任务与采样点数

        # 测试集（关闭数据增强）
        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')  # 使用 trainval 作为训练集

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))  # 训练集 DataLoader

    # 测试集 DataLoader
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)  # 类别数（如 ModelNet40 为 40）
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # -----------------------
    # 模型与优化器/调度器
    # -----------------------
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))  # 可加载预训练继续训练

    # 优化器与学习率调度
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize  # 仅用于日志打印

    # -----------------------
    # 循环训练与在线验证
    # -----------------------
    for epoch in range(opt.nepoch):
        scheduler.step()  # 更新学习率（若严格遵循 PyTorch 1.1+ 建议，可移至 epoch 末尾）
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]  # 部分数据集 target 形状为 (B,1)，取第 0 列
            points = points.transpose(2, 1)  # 维度转换 [B, N, 3] -> [B, 3, N]
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()  # 训练模式
            pred, trans, trans_feat = classifier(points)  # 前向传播，pred 为 log_softmax 概率
            loss = F.nll_loss(pred, target)  # NLL 损失
            if opt.feature_transform:  # 特征变换正则项（可提高旋转/尺度鲁棒性）
                loss += feature_transform_regularizer(trans_feat) * 0.001
            # 反向传播
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            # 每10个batch验证一次
            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()  # 评估模式
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
        # 保存周期模型
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    # -----------------------
    # 最终测试集完整评估
    # -----------------------
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()  # 评估模式
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
