from __future__ import print_function

# =============================================
# PointNet 语义分割训练脚本（ShapeNetPart 单类分割）
#
# 主要流程：
# 1) 解析命令行参数
# 2) 构建训练/测试数据集与 DataLoader
# 3) 初始化 PointNet 分割模型与优化器/学习率调度器
# 4) 迭代训练，每个 epoch 结束保存模型并在验证集上评估
# 5) 最后计算类内平均 IoU(mIOU)
#
# 重要说明（Windows 用户必读）：
# - Windows 平台的多进程 DataLoader 需要将主程序逻辑放入
#   if __name__ == '__main__': 保护块中，避免子进程启动报错。
# - 如果仍遇到多进程相关问题，可将 --workers 设置为 0 退回单进程加载。
# =============================================

# 点云分割训练脚本
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 将项目根目录加入 sys.path，便于绝对导入 pointnet 包

# 点云分割训练脚本
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    # -----------------------
    # 1) 命令行参数解析
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')  # 训练/测试批大小
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)  # DataLoader 的工作进程数（Windows 出错可设为 0）
    parser.add_argument(
        '--nepoch', type=int, default=5, help='number of epochs to train for')  # 训练轮数
    parser.add_argument('--outf', type=str, default='seg', help='output folder')  # 模型权重输出目录
    parser.add_argument('--model', type=str, default='', help='model path')  # 预训练模型权重路径（可选）
    parser.add_argument('--dataset', type=str, help="dataset path",
                        default="D:\\Code\\Python\\pointnet.pytorch-master\\shapenetcore_partanno_segmentation_benchmark_v0")  # ShapeNetPart 根目录
    parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")  # 训练/评估的单一类别（如 Chair）
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")  # 是否启用特征变换正则（提高鲁棒性）

    opt = parser.parse_args()
    print(opt)

    # 固定随机种子，保证可复现
    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # -----------------------
    # 2) 数据集与 DataLoader
    # -----------------------
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice])  # 指定为分割任务，并限定一个类别
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))  # 训练集 DataLoader

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)  # 测试集不做数据增强
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))  # 测试集 DataLoader

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes  # 当前类别对应的零件（part）种类数
    print('classes', num_classes)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'  # 终端蓝色高亮输出

    # -----------------------
    # 3) 模型与优化器/调度器
    # -----------------------
    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)  # k 即 part 类别数

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))  # 可加载预训练权重继续训练/评估

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每 20 个 epoch 将学习率乘以 0.5
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize  # 仅用于日志打印

    # -----------------------
    # 4) 训练与在线验证
    # -----------------------
    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            # 输入维度转换为 (B, 3, N)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()  # 进入训练模式
            pred, trans, trans_feat = classifier(points)
            # PointNetDenseCls 输出 (B, k, N)，拉平成 (B*N, k) 以计算逐点 NLL 损失
            pred = pred.view(-1, num_classes)
            # 标签原始范围为 [1..k]，这里减 1 转为 [0..k-1]
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)  # 逐点负对数似然损失
            if opt.feature_transform:  # 可选的特征变换正则项，鼓励更稳定的特征空间
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            # 统计逐点精度（正确点数 / 总点数）
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * 2500)))

            if i % 10 == 0:
                # 在线验证：每 10 个 batch 在测试集抽样一批进行评估
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()  # 切换到评估模式
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize * 2500)))

        # step the LR scheduler after optimizer.step() calls in the epoch
        scheduler.step()  # 注意：PyTorch 1.1+ 建议先 optimizer.step() 再 scheduler.step()

        # 保存当前 epoch 的模型权重
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

    # -----------------------
    # 5) 计算 mIOU（类内平均 IoU）
    # -----------------------
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]  # 对每个点取概率最大的 part 类别

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1  # 标签转为 [0..k-1]

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # 该类别下所有 part 索引
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # 若 GT 与预测都没有该 part，则该 part 视为 IoU=1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
