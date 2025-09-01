from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

"""3D空间变换网络，学习点云的空间变换矩阵"""


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 输入通道3（xyz坐标），输出64维特征
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 输入64维特征，输出128维特征
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 输入128维特征，输出1024维特征
        self.fc1 = nn.Linear(1024, 512)  # 输入1024维特征，输出512维特征
        self.fc2 = nn.Linear(512, 256)  # 输入512维特征，输出256维特征
        self.fc3 = nn.Linear(256, 9)  # 输入256维特征，输出9维特征（3x3矩阵）
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)  # 批归一化
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 3, N] -> [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, N] -> [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N] -> [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, N] -> [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024, 1] -> [B, 1024]
        x = F.relu(self.bn4(self.fc1(x)))  # [B, 1024] -> [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))  # [B, 512] -> [B, 256]
        x = self.fc3(x)  # [B, 256] -> [B, 9]

        # 恒等矩阵初始化
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)  # 变形为3x3矩阵 [B, 3, 3]
        return x


"""64维空间变换网络，学习点云的空间变换矩阵"""


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)  # 输入通道k（特征维度），输出64维特征
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 输入64维特征，输出128维特征
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 输入128维特征，输出1024维特征
        self.fc1 = nn.Linear(1024, 512)  # 输入1024维特征，输出512维特征
        self.fc2 = nn.Linear(512, 256)  # 输入512维特征，输出256维特征
        self.fc3 = nn.Linear(256, k * k)  # 输入256维特征，输出k*k维特征（k维空间变换矩阵）
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)  # 批归一化
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 3, N] -> [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, N] -> [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N] -> [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, N] -> [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024, 1] -> [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))  # [B, 1024] -> [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))  # [B, 512] -> [B, 256]
        x = self.fc3(x)  # [B, 256] -> [B, k*k]
        x = self.fc3(x)  # [B, k*k] -> [B, k*k]
        # 恒等矩阵初始化
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)  # [B, k*k] -> [B, k, k]
        return x


"""PointNet特征提取核心模块"""


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()  # 空间变换网络
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 输入通道3（xyz坐标），输出64维特征
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 输入64维特征，输出128维特征
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 输入128维特征，输出1024维特征
        self.bn1 = nn.BatchNorm1d(64)  # 批归一化
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:  # 如果使用特征变换
            self.fstn = STNkd(k=64)  # 空间变换网络

    def forward(self, x):
        n_pts = x.size()[2]  # 获取点云中的点数 输入：[B, N, 3]，其中 B 是批次大小，N 是点云中点的数量，3 是每个点的坐标维度（x,y,z）
        trans = self.stn(x)  # 通过 STN3d 空间变换网络计算空间变换矩阵
        x = x.transpose(2, 1)  # 维度转换 [B, N, 3] -> [B, 3, N]
        x = torch.bmm(x, trans)  # 对点进行空间变换，规范对齐
        x = x.transpose(2, 1)  # 恢复形状 [B, 3, N] -> [B, N, 3]
        x = F.relu(self.bn1(self.conv1(x)))  # 通过第一个卷积层提取初级特征，特征维度从 3 提升到 64， [B, N, 3] -> [B, 64, N]

        if self.feature_transform:  # 执行特征空间的变换
            trans_feat = self.fstn(x)  # 计算 64 维特征空间的变换矩阵
            x = x.transpose(2, 1)  # 维度转换 [B, N, 64] -> [B, 64, N]
            x = torch.bmm(x, trans_feat)  # 对点的特征进行空间变换，规范对齐
            x = x.transpose(2, 1)  # 恢复形状 [B, 64, N] -> [B, N, 64]
        else:
            trans_feat = None

        pointfeat = x  # 保存原始点特征
        x = F.relu(self.bn2(self.conv2(x)))  # 通过第二个卷积层进一步提取中级特征，特征维度从 64 提升到 128，[B, N, 64] -> [B, 128, N]
        x = self.bn3(self.conv3(x))  # 通过第三个卷积层提取高级特征，特征维度从 128 提升到 1024，[B, 128, N] -> [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # 执行全局最大池化，提取全局特征向量，[B, 1024, N] -> [B, 1024, 1]
        x = x.view(-1, 1024)  # 全局最大池化后，将特征维度从 1024 压缩到 1，[B, 1024, 1] -> [B, 1024]
        if self.global_feat:  # 如果是全局特征提取模式
            return x, trans, trans_feat  # 返回全局特征向量、空间变换矩阵和特征变换矩阵 
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)  # 将全局特征 x 复制扩展到每个点上，[B, 1024, 1] -> [B, 1024, N]
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # 拼接全局特征和点特征，[B, 1088, N]


"""PointNet分类网络"""


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform  # 是否使用特征变换
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)  # 特征提取网络，只返回全局特征向量
        self.fc1 = nn.Linear(1024, 512)  # 全连接层1，输入1024维特征，输出512维特征
        self.fc2 = nn.Linear(512, 256)  # 全连接层2，输入512维特征，输出256维特征
        self.fc3 = nn.Linear(256, k)  # 全连接层3，输入256维特征，输出k维特征（分类数） 
        self.dropout = nn.Dropout(p=0.3)  # 防止过拟合，随机将输入的30%元素设置为0
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化，对每个样本的每个特征维度进行归一化
        self.bn2 = nn.BatchNorm1d(256)  # 批归一化，对每个样本的每个特征维度进行归一化
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # 特征提取，返回全局特征向量
        # 通过全连接层提取高级特征，全连接层能够有效地处理这种全局特征，通过权重矩阵将高维特征转换为类别概率分布。
        x = F.relu(self.bn1(self.fc1(x)))  # 全连接层1，输入1024维特征，输出512维特征，[B, 1024] -> [B, 512]
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # 全连接层2，输入512维特征，输出256维特征，[B, 512] -> [B, 256]
        x = self.fc3(x)  # 全连接层3，输入256维特征，输出k维特征（分类数），[B, 256] -> [B, k]
        return F.log_softmax(x, dim=1), trans, trans_feat  # 应用 log_softmax 激活函数并返回最终结果，[B, k]


"""PointNet分割网络"""


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)  # 特征提取网络，返回每个点的特征向量
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)  # 输入通道1088（1024+64），输出512维特征
        self.conv2 = torch.nn.Conv1d(512, 256, 1)  # 输入512维特征，输出256维特征
        self.conv3 = torch.nn.Conv1d(256, 128, 1)  # 输入256维特征，输出128维特征
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)  # 输入128维特征，输出k维特征
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]  # 批次大小
        n_pts = x.size()[2]  # 点云采样数
        x, trans, trans_feat = self.feat(x)  # 特征提取，返回每个点的特征向量，1088（1024 全局特征 + 64 局部特征）
        # 通过一系列卷积层(共享权重)逐步降低特征维度，最终映射到分割类别数
        x = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积层，输入1088维特征，输出512维特征，[B, N, 1088] -> [B, 512, N]
        x = F.relu(self.bn2(self.conv2(x)))  # 第二个卷积层，输入512维特征，输出256维特征，[B, 512, N] -> [B, 256, N]
        x = F.relu(self.bn3(self.conv3(x)))  # 第三个卷积层，输入256维特征，输出128维特征，[B, 256, N] -> [B, 128, N]
        x = self.conv4(x)  # 第四个卷积层，输入128维特征，输出k维特征（分类数），[B, 128, N] -> [B, k, N]
        x = x.transpose(2, 1).contiguous()  # 维度转换 [B, k, N] -> [B, N, k]
        x = F.log_softmax(x.view(-1, self.k), dim=-1)  # 应用 log_softmax 激活函数，[B, N, k] -> [B*N, k]
        x = x.view(batchsize, n_pts, self.k)  # 恢复原始维度 [B*N, k] -> [B, N, k]
        return x, trans, trans_feat  # 返回每个点的分类结果、空间变换矩阵和特征变换矩阵


"""特征变换矩阵正则化器，约束变换矩阵正交性"""


def feature_transform_regularizer(trans):
    d = trans.size()[1]  # 特征维度
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]  # 单位矩阵 [1, d, d]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))  # 计算变换矩阵与单位矩阵的距离
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
