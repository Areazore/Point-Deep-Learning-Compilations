"""
Author: Benny
Date: Nov 2019

这个文件实现了 PointNet++ 模型在 ShapeNet 数据集上进行部件分割的测试代码。
主要功能包括：
1. 加载预训练的模型
2. 在测试集上评估模型性能
3. 计算每个类别的 IoU (交并比) 指标
4. 支持多次投票以提高预测准确性
"""

# 导入所需的Python库
import argparse  # 用于解析命令行参数
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset  # 导入ShapeNet数据集加载器
import torch
import logging  # 用于日志记录
import sys
import importlib  # 用于动态导入模型
from tqdm import tqdm  # 用于显示进度条
import numpy as np

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 定义ShapeNet数据集中各类别对应的部件标签
# 例如：'Earphone'类别包含标签16,17,18三个部件
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

# 创建标签到类别的映射字典
# 例如：标签0映射到'Airplane'，标签1映射到'Airplane'，以此类推
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """将标签转换为one-hot编码
    
    Args:
        y: 输入的标签张量
        num_classes: 类别总数
        
    Returns:
        one-hot编码后的张量，如果输入在GPU上则返回值也在GPU上
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]  # 创建one-hot编码
    if (y.is_cuda):
        return new_y.cuda()  # 如果输入在GPU上，将结果转移到GPU
    return new_y


def parse_args():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser('PointNet')
    # 设置测试时的批量大小
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    # 指定使用的GPU设备
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # 每个点云的点数量
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    # 实验结果保存路径
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    # 是否使用法向量信息
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # 测试时的投票次数，通过多次预测取平均来提高准确性
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        """日志记录函数"""
        logger.info(str)
        print(str)

    '''超参数设置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置使用的GPU
    experiment_dir = 'log/part_seg/' + args.log_dir  # 设置实验结果保存目录

    '''日志记录设置'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 创建日志文件
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # 数据集根目录
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # 加载测试数据集
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16  # 类别总数
    num_part = 50  # 部件总数

    '''模型加载'''
    # 从日志目录中获取模型名称
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # 动态导入模型
    MODEL = importlib.import_module(model_name)
    # 初始化模型并移动到GPU
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # 加载预训练的模型权重
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 在评估模式下进行测试
    with torch.no_grad():
        test_metrics = {}  # 存储测试指标
        total_correct = 0  # 总正确预测数
        total_seen = 0  # 总点数
        total_seen_class = [0 for _ in range(num_part)]  # 每个部件类别的点数
        total_correct_class = [0 for _ in range(num_part)]  # 每个部件类别正确预测的点数
        shape_ious = {cat: [] for cat in seg_classes.keys()}  # 存储每个类别的IoU
        seg_label_to_cat = {}  # 标签到类别的映射字典 {0:Airplane, 1:Airplane, ...49:Table}

        # 构建标签到类别的映射
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        # 设置模型为评估模式
        classifier = classifier.eval()
        
        # 遍历测试数据集
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            # 将数据转移到GPU
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # 转置点云数据以适应网络输入格式
            # 初始化投票池
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            # 多次投票以提高预测准确性
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            # 计算平均预测结果
            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            # 对每个样本进行预测
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]  # 获取当前样本的类别
                logits = cur_pred_val_logits[i, :, :]
                # 根据类别选择相应的部件标签范围进行预测
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            # 计算正确预测的点数
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            # 统计每个部件类别的预测情况
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)  # 该部件类别的总点数
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # 该部件类别正确预测的点数

            # 计算每个样本的IoU
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]  # 预测的部件标签
                segl = target[i, :]  # 真实的部件标签
                cat = seg_label_to_cat[segl[0]]  # 获取类别
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # 存储每个部件的IoU
                # 计算每个部件的IoU
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # 如果该部件不存在且没有被预测
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:  # 计算IoU：交集/并集
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))  # 计算该样本所有部件的平均IoU

        # 计算所有类别的平均IoU
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])  # 计算每个类别的平均IoU
        mean_shape_ious = np.mean(list(shape_ious.values()))  # 计算所有类别的平均IoU

        # 记录各项评估指标
        test_metrics['accuracy'] = total_correct / float(total_seen)  # 总体准确率
        test_metrics['class_avg_accuracy'] = np.mean(  # 类别平均准确率
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        # 输出每个类别的IoU
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious  # 类别平均IoU
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)  # 实例平均IoU

    # 输出最终评估结果
    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])  # 总体准确率
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])  # 类别平均准确率
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])  # 类别平均IoU
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])  # 实例平均IoU


if __name__ == '__main__':
    args = parse_args()
    main(args)
