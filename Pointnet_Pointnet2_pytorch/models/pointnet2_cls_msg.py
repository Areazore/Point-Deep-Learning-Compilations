# 导入必要的库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction  # 导入点云集合抽象层模块

# PointNet++ MSG（Multi-Scale Grouping）模型用于点云分类任务
# MSG表示多尺度分组，通过使用不同半径的球查询来捕获不同尺度的特征


class get_model(nn.Module):
    """PointNet++ MSG（Multi-Scale Grouping）分类模型
    
    该模型使用多尺度特征提取策略，通过不同半径的球查询来捕获不同尺度的局部特征
    """
    def __init__(self, num_class, normal_channel=True):
        """初始化PointNet++ MSG分类模型
        
        参数:
            num_class: 分类类别数量
            normal_channel: 是否使用法向量作为额外输入特征，如果为True，则输入通道数为3+3=6
        """
        super(get_model, self).__init__()
        
        # 确定输入通道数，如果使用法向量则为3，否则为0（坐标本身不算在通道数中）
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        # 第一层多尺度集合抽象层
        # 采样512个点，使用3个不同半径[0.1, 0.2, 0.4]的球查询，每个球内采样[16, 32, 128]个点
        # 对应的MLP配置为[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        # 第二层多尺度集合抽象层
        # 采样128个点，使用3个不同半径[0.2, 0.4, 0.8]的球查询，每个球内采样[32, 64, 128]个点
        # 输入通道数为320（第一层输出的64+128+128=320个通道）
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        # 第三层全局集合抽象层
        # 不进行采样(None)，处理所有点，输入通道数为640+3（第二层输出的128+256+256=640个通道加上坐标3个通道）
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        # 全连接层进行分类
        self.fc1 = nn.Linear(1024, 512)  # 第一个全连接层
        self.bn1 = nn.BatchNorm1d(512)   # 批归一化
        self.drop1 = nn.Dropout(0.4)     # dropout防止过拟合
        
        self.fc2 = nn.Linear(512, 256)   # 第二个全连接层
        self.bn2 = nn.BatchNorm1d(256)   # 批归一化
        self.drop2 = nn.Dropout(0.5)     # dropout防止过拟合
        
        self.fc3 = nn.Linear(256, num_class)  # 最终分类层

    def forward(self, xyz):
        """前向传播函数
        
        参数:
            xyz: 输入点云数据，形状为[B, C, N]，其中B为批次大小，C为通道数（3或6，取决于是否使用法向量），
                 N为点数量
        返回:
            x: 分类预测结果，形状为[B, num_class]
            l3_points: 全局特征，形状为[B, 1024]
        """
        B, _, _ = xyz.shape  # 获取批次大小
        
        # 如果使用法向量，则分离坐标和法向量
        if self.normal_channel:
            norm = xyz[:, 3:, :]  # 法向量部分
            xyz = xyz[:, :3, :]   # 坐标部分
        else:
            norm = None

        # 通过三层特征提取网络
        l1_xyz, l1_points = self.sa1(xyz, norm)  # 第一层MSG，输出512个点的特征
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # 第二层MSG，输出128个点的特征
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # 第三层全局特征，输出1个点的特征

        # 将全局特征展平
        x = l3_points.view(B, 1024)  # [B, 1024]
        
        # 通过全连接层进行分类
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))  # 第一个全连接层+批归一化+ReLU+Dropout
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))  # 第二个全连接层+批归一化+ReLU+Dropout
        x = self.fc3(x)  # 最终分类层
        
        # 使用log_softmax获得对数概率
        x = F.log_softmax(x, -1)

        return x, l3_points  # 返回分类结果和全局特征


class get_loss(nn.Module):
    """损失函数类
    
    计算分类任务的损失函数
    """
    def __init__(self):
        """初始化损失函数类"""
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        """前向传播函数
        
        参数:
            pred: 模型预测结果，形状为[B, num_class]，包含log_softmax后的对数概率
            target: 目标标签，形状为[B]，包含类别索引
            trans_feat: 特征变换矩阵，在MSG模型中未使用
        返回:
            total_loss: 分类损失值
        """
        # 使用负对数似然损失(NLL Loss)计算分类损失
        # 由于pred已经过log_softmax处理，直接使用nll_loss
        total_loss = F.nll_loss(pred, target)

        return total_loss


