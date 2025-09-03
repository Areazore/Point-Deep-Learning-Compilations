"""
@Author: Benny
@Date: 2019-11
@Description: 用于S3DIS数据集的语义分割测试
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene  # S3DIS数据集加载器
from data_utils.indoor3d_util import g_label2color  # 标签到颜色的映射
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录
ROOT_DIR = BASE_DIR
# 将模型目录添加到系统路径中
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 定义S3DIS数据集的类别
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
# 创建类别到标签的映射
class2label = {cls: i for i, cls in enumerate(classes)}
# 分割类别
seg_classes = class2label
# 创建标签到类别的映射
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='测试时的批处理大小 [默认: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备 [默认: 0]')
    parser.add_argument('--num_point', type=int, default=4096, help='点数 [默认: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='实验根目录')
    parser.add_argument('--visual', action='store_true', default=False, help='可视化结果 [默认: False]')
    parser.add_argument('--test_area', type=int, default=5, help='用于测试的区域, 可选: 1-6 [默认: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='通过投票聚合分割分数 [默认: 3]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """将预测标签添加到投票池中

    Args:
        vote_label_pool: 投票池, (N, num_classes)
        point_idx: 点的索引, (B, npoint)
        pred_label: 预测的标签, (B, npoint)
        weight: 权重, (B, npoint)
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        """记录日志并打印"""
        logger.info(str)
        print(str)

    """超参数设置"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置GPU
    experiment_dir = 'log/sem_seg/' + args.log_dir  # 实验目录
    visual_dir = experiment_dir + '/visual/'  # 可视化结果保存目录
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)  # 创建可视化目录

    """日志设置"""
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 日志文件
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13  # 类别数
    BATCH_SIZE = args.batch_size  # 批处理大小
    NUM_POINT = args.num_point  # 点数

    root = 'data/s3dis/stanford_indoor3d/'  # 数据集根目录

    # 加载S3DIS测试数据集
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    """模型加载"""
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]  # 获取模型名称
    MODEL = importlib.import_module(model_name)  # 动态导入模型
    classifier = MODEL.get_model(NUM_CLASSES).cuda()  # 实例化模型并移动到GPU
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  # 加载预训练权重
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()  # 设置为评估模式

    with torch.no_grad():
        # 获取场景ID列表
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]  # 去除文件扩展名
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        # 初始化评估指标
        total_seen_class = [0 for _ in range(NUM_CLASSES)]  # 每个类别的总点数
        total_correct_class = [0 for _ in range(NUM_CLASSES)]  # 每个类别正确预测的点数
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]  # 每个类别IoU的分母

        log_string('---- EVALUATION WHOLE SCENE----')

        # 遍历每个场景
        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            # 初始化当前场景的评估指标
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]  # 当前场景每个类别的总点数
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]  # 当前场景每个类别正确预测的点数
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]  # 当前场景每个类别IoU的分母

            # 如果需要可视化，创建预测和真实标签的obj文件
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            # 获取整个场景的点云数据和标签
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            # 初始化投票池
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))

            # 多次投票以提高预测准确性
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                # 获取当前场景的数据块
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                # 按批次处理数据块
                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    # 填充当前批次的数据
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0  # 归一化RGB值

                    # 转换数据格式并进行预测
                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)  # 转置以适应网络输入格式
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    # 将预测结果添加到投票池
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            # 根据投票结果获取最终预测标签
            pred_label = np.argmax(vote_label_pool, 1)

            # 计算每个类别的评估指标
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))  # 该类别的总点数
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))  # 该类别正确预测的点数
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))  # 该类别IoU的分母
                # 累加到全局指标
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            # 计算当前场景的IoU
            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])  # 计算非空类别的平均IoU
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            # 保存预测结果
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()

            # 如果需要可视化，保存带颜色的点云
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]  # 预测标签对应的颜色
                color_gt = g_label2color[whole_scene_label[i]]  # 真实标签对应的颜色
                if args.visual:
                    # 写入预测结果的点云
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    # 写入真实标签的点云
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()

        # 计算所有场景的平均IoU
        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        # 输出每个类别的IoU
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)

        # 输出整体评估指标
        log_string('eval point avg class IoU: %f' % np.mean(IoU))  # 类别平均IoU
        log_string('eval whole scene point avg class acc: %f' % (  # 类别平均准确率
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (  # 整体准确率
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
