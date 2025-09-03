"""
Author: Benny
Date: Nov 2019
点云分类测试脚本 - 用于测试训练好的PointNet/PointNet++等模型在点云分类任务上的性能
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader  # 数据加载器
import argparse  # 命令行参数解析
import numpy as np
import os
import torch
import logging  # 日志记录
from tqdm import tqdm  # 进度条显示
import sys
import importlib  # 用于动态导入模型

# 设置基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 将模型目录添加到系统路径中，以便导入模型模块


def parse_args():
    '''参数设置'''
    parser = argparse.ArgumentParser('Testing')
    # 硬件相关参数
    parser.add_argument('--use_cpu', action='store_true', default=False, help='使用CPU模式进行测试')
    parser.add_argument('--gpu', type=str, default='0', help='指定使用的GPU设备编号')
    # 测试相关参数
    parser.add_argument('--batch_size', type=int, default=24, help='测试时的批次大小')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='测试使用的ModelNet数据集类别数(10或40)')
    parser.add_argument('--num_point', type=int, default=1024, help='每个点云的点数量')
    parser.add_argument('--log_dir', type=str, required=True, help='实验结果目录，必须指定')
    parser.add_argument('--use_normals', action='store_true', default=False, help='是否使用法向量特征')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='是否使用均匀采样')
    parser.add_argument('--num_votes', type=int, default=3, help='使用投票机制聚合分类得分的投票次数')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    '''
    测试模型在测试集上的性能，支持多次投票机制
    
    参数:
        model: 待测试的模型
        loader: 测试数据加载器
        num_class: 类别数量，默认为40
        vote_num: 投票次数，默认为1，增加可提高准确率
        
    返回:
        instance_acc: 实例准确率（每个点云被正确分类的比例）
        class_acc: 类别准确率（每个类别的平均准确率）
    '''
    mean_correct = []  # 存储每个批次的准确率
    classifier = model.eval()  # 将模型设置为评估模式
    class_acc = np.zeros((num_class, 3))  # 存储每个类别的准确率统计信息

    # 遍历测试数据集
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # 如果使用GPU，将数据转移到GPU
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  # 转置点云数据形状以适应模型输入 [B,N,C] -> [B,C,N]
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()  # 创建投票池，用于存储多次预测结果

        # 多次投票预测
        for _ in range(vote_num):
            pred, _ = classifier(points)  # 前向传播得到预测结果
            vote_pool += pred  # 累加预测结果到投票池
        pred = vote_pool / vote_num  # 取平均得到最终预测结果
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
    主函数 - 测试模型性能的主要流程
    
    参数:
        args: 解析的命令行参数
    '''
    def log_string(str):
        '''日志记录函数，同时输出到控制台和日志文件'''
        logger.info(str)
        print(str)

    '''设置超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的GPU设备

    '''确定实验目录'''
    experiment_dir = 'log/classification/' + args.log_dir  # 实验结果目录路径

    '''设置日志系统'''
    args = parse_args()  # 解析命令行参数
    # 创建日志记录器
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 记录测试参数
    log_string('PARAMETER ...')
    log_string(args)

    '''加载数据集'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'  # ModelNet40数据集路径

    # 创建测试集
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    # 创建测试数据加载器
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,  # 批次大小
        shuffle=False,  # 测试集不需要打乱
        num_workers=10  # 多进程加载数据的进程数
    )

    '''加载模型'''
    num_class = args.num_category  # 类别数量
    # 从日志目录中获取模型名称
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # 动态导入指定的模型模块
    model = importlib.import_module(model_name)

    # 创建分类器模型实例
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # 如果使用GPU，将模型转移到GPU
    if not args.use_cpu:
        classifier = classifier.cuda()

    # 加载训练好的模型参数
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上评估模型性能
    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 使用投票机制测试模型性能
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        # 记录测试结果
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    '''
    程序入口点
    '''
    args = parse_args()  # 解析命令行参数
    main(args)  # 调用主函数开始测试
