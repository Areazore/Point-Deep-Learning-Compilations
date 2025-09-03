import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

# 这个文件实现了PointNet++用于语义分割的多尺度分组(MSG)模型
# MSG模型使用多个半径进行特征提取，能够更好地捕获不同尺度的几何信息
#
# 四个模型的区别：
# 1. 任务类型区别：
#    - 语义分割模型(pointnet2_sem_seg.py和pointnet2_sem_seg_msg.py)：为点云中每个点分配语义类别
#    - 部件分割模型(pointnet2_part_seg_ssg.py和pointnet2_part_seg_msg.py)：为点云中每个点分配部件类别，需要额外的形状类别信息
# 2. 特征提取方式区别：
#    - 单尺度分组(SSG)模型：每层使用单一半径进行特征提取，计算效率高
#    - 多尺度分组(MSG)模型：每层使用多个半径进行特征提取，能更好捕获不同尺度几何信息
# 3. 架构差异：
#    - 语义分割模型：使用4个下采样层和4个上采样层
#    - 部件分割模型：使用3个下采样层和3个上采样层，需要额外的形状类别信息


class get_model(nn.Module):
    def __init__(self, num_classes):
        """初始化语义分割MSG模型
        
        参数:
            num_classes: 分割类别数量
        """
        super(get_model, self).__init__()

        # 设置多尺度抽象层 (Set Abstraction MSG layers) - 每层使用多个半径进行特征提取
        # 参数依次为: npoint(采样点数), radius_list(多个球半径), nsample_list(每个半径对应的邻居数量), 
        # in_channel(输入特征维度), mlp_list(每个尺度对应的多层感知机通道数)
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])  # 第一层SA: 1024个中心点, 两个尺度
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])  # 第二层SA: 256个中心点, 两个尺度
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])  # 第三层SA: 64个中心点, 两个尺度
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])  # 第四层SA: 16个中心点, 两个尺度
        
        # 特征传播层 (Feature Propagation layers) - 上采样并传播特征
        # 参数为: in_channel(输入特征维度), mlp(多层感知机通道数)
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])  # 第四层FP: 从16点传播到64点
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])  # 第三层FP: 从64点传播到256点
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])  # 第二层FP: 从256点传播到1024点
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])  # 第一层FP: 从1024点传播到原始点数
        
        # 最终分类层
        self.conv1 = nn.Conv1d(128, 128, 1)  # 1x1卷积层
        self.bn1 = nn.BatchNorm1d(128)  # 批归一化
        self.drop1 = nn.Dropout(0.5)  # dropout防止过拟合
        self.conv2 = nn.Conv1d(128, num_classes, 1)  # 最终分类层, 输出num_classes个通道

    def forward(self, xyz):
        """前向传播函数
        
        参数:
            xyz: 输入点云数据, 形状为(B, C, N), 其中B是批量大小, C是通道数, N是点数量
                前3个通道为xyz坐标, 后面可能包含其他特征如法线、颜色等
                
        返回:
            x: 每个点的分类结果, 形状为(B, N, num_classes)
            l4_points: 全局特征, 用于其他任务
        """
        # 初始化第0层点和特征
        l0_points = xyz  # 原始点特征
        l0_xyz = xyz[:,:3,:]  # 提取xyz坐标

        # 多尺度设置抽象(SA)层: 逐步下采样并提取多尺度特征
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # 1024个点, 多尺度特征
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # 256个点, 多尺度特征
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # 64个点, 多尺度特征
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # 16个点, 多尺度特征

        # 特征传播(FP)层: 上采样并传播特征 - 从粗到细
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # 16点 -> 64点
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # 64点 -> 256点
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # 256点 -> 1024点
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # 1024点 -> 原始点数

        # 最终分类层
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # 特征转换
        x = self.conv2(x)  # 分类预测
        x = F.log_softmax(x, dim=1)  # 对数softmax
        x = x.permute(0, 2, 1)  # 调整维度顺序为(B, N, C)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        """初始化损失函数类"""
        super(get_loss, self).__init__()
        
    def forward(self, pred, target, trans_feat, weight):
        """计算损失函数
        
        参数:
            pred: 预测结果, 形状为(B, N, num_classes)
            target: 目标标签, 形状为(B, N)
            trans_feat: 变换矩阵特征(在此模型中未使用)
            weight: 各类别的权重
            
        返回:
            total_loss: 负对数似然损失
        """
        # 使用带权重的负对数似然损失函数
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    # 测试代码
    import  torch
    model = get_model(13)  # 创建13个类别的语义分割模型
    xyz = torch.rand(6, 9, 2048)  # 创建随机点云数据: 批量大小为6, 9个通道, 每个点云2048个点
    (model(xyz))  # 测试模型前向传播