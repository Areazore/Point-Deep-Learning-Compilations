import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

# 这个文件实现了PointNet++用于部件分割的多尺度分组(MSG)模型
# MSG模型使用多个半径进行特征提取，能够更好地捕获不同尺度的几何信息
# 部件分割任务是为点云中的每个点分配一个部件类别标签，如椅子的腿、座位、靠背等
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
    def __init__(self, num_classes, normal_channel=False):
        """初始化部件分割MSG模型
        
        参数:
            num_classes: 分割类别数量
            normal_channel: 是否使用法向量作为额外特征
        """
        super(get_model, self).__init__()
        # 根据是否使用法向量特征设置额外通道数
        if normal_channel:
            additional_channel = 3  # 如果使用法向量，增加3个通道
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        # 设置多尺度抽象层 (Set Abstraction MSG layers) - 每层使用多个半径进行特征提取
        # 参数依次为: npoint(采样点数), radius_list(多个球半径), nsample_list(每个半径对应的邻居数量), 
        # in_channel(输入特征维度), mlp_list(每个尺度对应的多层感知机通道数)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, 
                                            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])  # 第一层SA: 512个中心点, 三个尺度
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, 
                                            [[128, 128, 256], [128, 196, 256]])  # 第二层SA: 128个中心点, 两个尺度
        
        # 第三层使用单尺度全局特征提取
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                         in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)  # 全局特征
        
        # 特征传播层 (Feature Propagation layers) - 上采样并传播特征
        # 参数为: in_channel(输入特征维度), mlp(多层感知机通道数)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])  # 第三层FP: 从全局传播到128点
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])  # 第二层FP: 从128点传播到512点
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])  # 第一层FP: 从512点传播到原始点数
        
        # 最终分类层
        self.conv1 = nn.Conv1d(128, 128, 1)  # 1x1卷积层
        self.bn1 = nn.BatchNorm1d(128)  # 批归一化
        self.drop1 = nn.Dropout(0.5)  # dropout防止过拟合
        self.conv2 = nn.Conv1d(128, num_classes, 1)  # 最终分类层, 输出num_classes个通道

    def forward(self, xyz, cls_label):
        """前向传播函数
        
        参数:
            xyz: 输入点云数据, 形状为(B, C, N), 其中B是批量大小, C是通道数(3或6), N是点数量
            cls_label: 形状类别标签, 形状为(B, 16), 16是one-hot编码的类别数
            
        返回:
            x: 每个点的部件分类预测结果, 形状为(B, N, num_classes)
            l3_points: 全局特征, 可用于其他任务
        """
        # 设置抽象层 (Set Abstraction layers)
        B,C,N = xyz.shape  # 获取批量大小、通道数和点数量
        if self.normal_channel:
            l0_points = xyz  # 如果使用法向量，则使用全部通道作为特征
            l0_xyz = xyz[:,:3,:]  # 只使用前3个通道(xyz坐标)作为位置信息
        else:
            l0_points = xyz  # 不使用法向量时，xyz既是位置也是特征
            l0_xyz = xyz
            
        # 多尺度特征提取过程
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # 第一层SA: 从N点降采样到512点
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # 第二层SA: 从512点降采样到128点
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # 第三层SA: 全局特征
        
        # 特征传播层 (Feature Propagation layers) - 上采样过程
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # 从全局特征传播到128点
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # 从128点传播到512点
        
        # 将形状类别信息转换为one-hot编码并与点特征连接
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)  # 将类别标签扩展到每个点
        # 将类别信息、坐标和原始特征连接，然后传播到所有点
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        
        # 全连接层进行最终分类
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # 1x1卷积+批归一化+ReLU
        x = self.drop1(feat)  # dropout防止过拟合
        x = self.conv2(x)  # 最终分类层
        x = F.log_softmax(x, dim=1)  # 对数softmax
        x = x.permute(0, 2, 1)  # 调整维度顺序为(B, N, num_classes)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        """初始化损失函数类"""
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        """计算损失函数
        
        参数:
            pred: 预测结果, 形状为(B, N, num_classes)
            target: 目标标签, 形状为(B, N)
            trans_feat: 变换矩阵特征(在此模型中未使用)
            
        返回:
            total_loss: 负对数似然损失
        """
        # 使用负对数似然损失函数
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ == '__main__':
    import  torch
    # 测试代码: 创建一个有13个类别的部件分割模型
    model = get_model(13, normal_channel=False)
    # 生成随机点云数据: 批量大小为6, 3个通道, 2048个点
    xyz = torch.rand(6, 3, 2048)
    # 生成随机形状类别标签: 批量大小为6, 16个类别(one-hot编码)
    label = torch.rand(6, 16)
    # 前向传播测试
    output, _ = model(xyz, label)
    # 打印输出形状, 应为[6, 2048, 13], 表示6个样本, 每个样本2048个点, 每个点13个类别的预测概率
    print(output.shape)