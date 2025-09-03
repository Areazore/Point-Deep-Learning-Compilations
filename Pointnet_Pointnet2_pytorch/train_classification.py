"""
Author: Benny
Date: Nov 2019
点云分类训练脚本 - 用于训练PointNet/PointNet++等模型进行3D点云分类
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider  # 点云数据增强和处理工具
import importlib  # 用于动态导入模型
import shutil
import argparse  # 命令行参数解析

from pathlib import Path  # 路径处理
from tqdm import tqdm  # 进度条显示
from data_utils.ModelNetDataLoader import ModelNetDataLoader  # 数据加载器

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 将模型目录添加到系统路径中，以便导入模型模块

def parse_args():
    '''参数设置'''
    parser = argparse.ArgumentParser('training')
    # 硬件相关参数
    parser.add_argument('--use_cpu', action='store_true', default=False, help='使用CPU模式进行训练')
    parser.add_argument('--gpu', type=str, default='0', help='指定使用的GPU设备编号')
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=24, help='训练时的批次大小')
    parser.add_argument('--model', default='pointnet_cls', help='模型名称 [默认: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='训练使用的ModelNet数据集类别数(10或40)')
    parser.add_argument('--epoch', default=200, type=int, help='训练的轮数')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='训练的学习率')
    parser.add_argument('--num_point', type=int, default=1024, help='每个点云的点数量')
    parser.add_argument('--optimizer', type=str, default='Adam', help='训练使用的优化器')
    parser.add_argument('--log_dir', type=str, default=None, help='实验结果保存的根目录')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='权重衰减率')
    # 数据处理相关参数
    parser.add_argument('--use_normals', action='store_true', default=False, help='是否使用法向量特征')
    parser.add_argument('--process_data', action='store_true', default=False, help='是否离线保存处理后的数据')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='是否使用均匀采样')
    return parser.parse_args()


def inplace_relu(m):
    '''
    将模型中的所有ReLU激活函数设置为inplace模式，以节省内存
    
    参数:
        m: 模型中的模块
    '''
    classname = m.__class__.__name__  # 获取模块的类名
    if classname.find('ReLU') != -1:  # 如果是ReLU激活函数
        m.inplace=True  # 设置为inplace模式，减少内存占用


def test(model, loader, num_class=40):
    '''
    测试模型在测试集上的性能
    
    参数:
        model: 待测试的模型
        loader: 测试数据加载器
        num_class: 类别数量，默认为40
        
    返回:
        instance_acc: 实例准确率（每个点云被正确分类的比例）
        class_acc: 类别准确率（每个类别的平均准确率）
    '''
    mean_correct = []  # 存储每个批次的准确率
    class_acc = np.zeros((num_class, 3))  # 存储每个类别的准确率统计信息
    classifier = model.eval()  # 将模型设置为评估模式

    # 遍历测试数据集
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # 如果使用GPU，将数据转移到GPU
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  # 转置点云数据形状以适应模型输入 [B,N,C] -> [B,C,N]
        pred, _ = classifier(points)  # 前向传播得到预测结果
        pred_choice = pred.data.max(1)[1]  # 获取预测的类别索引

        # 计算每个类别的准确率
        for cat in np.unique(target.cpu()):
            # 计算当前类别的正确预测数量
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            # 累加当前类别的准确率
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1  # 累加当前类别的样本数量

        # 计算当前批次的准确率
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # 计算每个类别的平均准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    # 计算整体实例准确率
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    '''
    主函数 - 训练和测试模型的主要流程
    
    参数:
        args: 解析的命令行参数
    '''
    def log_string(str):
        '''日志记录函数，同时输出到控制台和日志文件'''
        logger.info(str)
        print(str)

    '''设置超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的GPU设备

    '''创建实验目录结构'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))  # 获取当前时间作为实验ID
    # 创建日志根目录
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    # 创建分类任务目录
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    # 创建具体实验目录（使用指定的log_dir或当前时间）
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # 创建模型检查点目录
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # 创建日志文件目录
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''设置日志系统'''
    args = parse_args()  # 解析命令行参数
    # 创建日志记录器
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 记录训练参数
    log_string('PARAMETER ...')
    log_string(args)

    '''加载数据集'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'  # ModelNet40数据集路径

    # 创建训练集和测试集
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    
    # 创建数据加载器
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,  # 批次大小
        shuffle=True,  # 随机打乱数据
        num_workers=10,  # 多进程加载数据的进程数
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 测试集不需要打乱
        num_workers=10
    )

    '''加载模型'''
    num_class = args.num_category  # 类别数量
    # 动态导入指定的模型模块
    model = importlib.import_module(args.model)
    # 复制模型文件到实验目录，便于复现
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    # 创建分类器模型实例
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # 获取损失函数
    criterion = model.get_loss()
    # 将模型中的ReLU设置为inplace模式以节省内存
    classifier.apply(inplace_relu)

    # 如果使用GPU，将模型和损失函数转移到GPU
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # 尝试加载预训练模型
    try:
        # 加载检查点文件
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']  # 获取已训练的轮数
        # 加载模型参数
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        # 如果没有找到预训练模型，从头开始训练
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # 配置优化器
    if args.optimizer == 'Adam':
        # 使用Adam优化器
        optimizer = torch.optim.Adam(
            classifier.parameters(),  # 优化模型参数
            lr=args.learning_rate,  # 学习率
            betas=(0.9, 0.999),  # Adam优化器的动量参数
            eps=1e-08,  # 数值稳定性参数
            weight_decay=args.decay_rate  # L2正则化系数
        )
    else:
        # 使用SGD优化器
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # 学习率调度器，每20个epoch将学习率乘以0.7
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # 初始化训练状态变量
    global_epoch = 0  # 全局轮数计数
    global_step = 0  # 全局步数计数
    best_instance_acc = 0.0  # 最佳实例准确率
    best_class_acc = 0.0  # 最佳类别准确率

    '''开始训练'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        # 记录当前训练轮次
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []  # 存储每个批次的准确率
        classifier = classifier.train()  # 将模型设置为训练模式

        # 更新学习率
        scheduler.step()
        
        # 遍历训练数据集
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()  # 清空梯度

            # 数据增强处理
            points = points.data.numpy()
            points = provider.random_point_dropout(points)  # 随机丢弃点
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  # 随机缩放
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  # 随机平移
            points = torch.Tensor(points)  # 转回PyTorch张量
            points = points.transpose(2, 1)  # 转置点云形状 [B,N,C] -> [B,C,N]

            # 如果使用GPU，将数据转移到GPU
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # 前向传播
            pred, trans_feat = classifier(points)  # 获取预测结果和特征变换矩阵
            # 计算损失
            loss = criterion(pred, target.long(), trans_feat)
            # 获取预测的类别
            pred_choice = pred.data.max(1)[1]

            # 计算准确率
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            global_step += 1  # 更新全局步数

        # 计算并记录训练集上的准确率
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # 在测试集上评估模型性能
        with torch.no_grad():  # 禁用梯度计算，节省内存
            # 测试模型性能
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            # 更新最佳实例准确率
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            # 更新最佳类别准确率
            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
            # 记录测试结果
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            # 如果当前模型性能最佳，保存模型
            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                # 保存模型状态
                state = {
                    'epoch': best_epoch,  # 最佳轮次
                    'instance_acc': instance_acc,  # 实例准确率
                    'class_acc': class_acc,  # 类别准确率
                    'model_state_dict': classifier.state_dict(),  # 模型参数
                    'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
                }
                torch.save(state, savepath)  # 保存模型
            global_epoch += 1  # 更新全局轮数

    logger.info('End of training...')  # 训练结束


if __name__ == '__main__':
    '''
    程序入口点
    '''
    args = parse_args()  # 解析命令行参数
    main(args)  # 调用主函数开始训练
