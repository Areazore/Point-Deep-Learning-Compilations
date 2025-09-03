# PointNet++工具函数和核心模块实现
import torch  # PyTorch深度学习库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from time import time  # 用于性能计时
import numpy as np  # 数值计算库

def timeit(tag, t):
    """计时函数，用于测量代码执行时间
    
    参数:
        tag: 标识符，用于标记当前计时的代码段
        t: 起始时间
        
    返回:
        当前时间，可用于下一段代码的计时
    """
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    """点云归一化函数
    
    将点云中心移到原点并缩放到单位球内
    
    参数:
        pc: 输入点云数据，形状为[N, C]，其中N为点数量，C为坐标维度
        
    返回:
        归一化后的点云数据
    """
    l = pc.shape[0]  # 获取点的数量
    centroid = np.mean(pc, axis=0)  # 计算点云中心点
    pc = pc - centroid  # 将点云中心移到原点
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 计算点到原点的最大距离
    pc = pc / m  # 缩放点云，使最远点的距离为1
    return pc

def square_distance(src, dst):
    """计算两组点之间的欧氏距离的平方

    计算公式推导:
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    参数:
        src: 源点集，形状为[B, N, C]，其中B为批次大小，N为源点数量，C为坐标维度
        dst: 目标点集，形状为[B, M, C]，其中B为批次大小，M为目标点数量，C为坐标维度
    返回:
        dist: 每对点之间的平方距离，形状为[B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # 计算-2 * (src点乘dst)部分
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # 加上src的平方和，扩展维度以便广播
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # 加上dst的平方和，扩展维度以便广播
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """根据索引获取点云中的点

    参数:
        points: 输入点云数据，形状为[B, N, C]，其中B为批次大小，N为点数量，C为点特征维度
        idx: 采样索引数据，形状为[B, S]或[B, S, K]等，其中B为批次大小，S为采样点数量
    返回:
        new_points: 索引后的点云数据，形状为[B, S, C]或[B, S, K, C]等
    """
    device = points.device  # 获取设备信息
    B = points.shape[0]  # 获取批次大小
    
    # 处理批次索引，以便正确索引每个批次中的点
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # 将除第一维外的所有维度设为1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 第一维设为1，用于重复
    
    # 创建批次索引并重复到与idx相同的形状
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    
    # 使用高级索引获取指定的点
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """最远点采样算法
    
    通过迭代选择距离当前点集最远的点，确保采样点在空间中均匀分布
    
    参数:
        xyz: 点云数据，形状为[B, N, 3]，其中B为批次大小，N为点数量
        npoint: 需要采样的点数量
    返回:
        centroids: 采样点的索引，形状为[B, npoint]
    """
    device = xyz.device  # 获取设备信息
    B, N, C = xyz.shape  # 获取批次大小、点数量和坐标维度
    
    # 初始化采样点索引和距离矩阵
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 存储采样点索引
    distance = torch.ones(B, N).to(device) * 1e10  # 初始化距离为一个很大的值
    
    # 随机选择初始点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 每个批次随机选择一个起始点
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 批次索引
    
    # 迭代选择最远点
    for i in range(npoint):
        centroids[:, i] = farthest  # 将当前最远点加入采样集合
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 获取当前中心点坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算所有点到当前中心点的距离
        mask = dist < distance  # 找出距离更小的点
        distance[mask] = dist[mask]  # 更新距离
        farthest = torch.max(distance, -1)[1]  # 选择距离最大的点作为下一个中心点
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询算法，查找每个中心点周围指定半径内的点
    
    参数:
        radius: 局部区域半径，决定了搜索范围
        nsample: 每个局部区域内的最大采样点数
        xyz: 所有点的坐标，形状为[B, N, 3]，其中B为批次大小，N为点数量
        new_xyz: 查询点（中心点）的坐标，形状为[B, S, 3]，其中S为查询点数量
    返回:
        group_idx: 分组后的点索引，形状为[B, S, nsample]，表示每个查询点周围的nsample个点的索引
    """
    device = xyz.device  # 获取设备信息
    B, N, C = xyz.shape  # 获取批次大小、点数量和坐标维度
    _, S, _ = new_xyz.shape  # 获取查询点数量
    
    # 创建索引矩阵，初始包含所有点的索引
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    
    # 计算查询点与所有点之间的平方距离
    sqrdists = square_distance(new_xyz, xyz)
    
    # 将距离大于半径平方的点的索引设为N（表示无效）
    group_idx[sqrdists > radius ** 2] = N
    
    # 对每个查询点的邻居按距离排序，并取前nsample个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 处理特殊情况：如果某个查询点周围的有效点少于nsample个，则用第一个有效点填充
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """采样并分组函数，实现PointNet++中的Set Abstraction操作的核心步骤
    
    首先通过最远点采样选择中心点，然后对每个中心点进行球查询找到邻域点，最后将邻域点特征组合
    
    参数:
        npoint: 采样点数量
        radius: 球查询的半径
        nsample: 每个局部区域内的最大采样点数
        xyz: 输入点的位置数据，形状为[B, N, 3]，其中B为批次大小，N为点数量
        points: 输入点的特征数据，形状为[B, N, D]，其中D为特征维度
        returnfps: 是否返回FPS采样的其他信息，默认为False
    返回:
        new_xyz: 采样后的点位置数据，形状为[B, npoint, 3]
        new_points: 采样后的点特征数据，形状为[B, npoint, nsample, 3+D]或[B, npoint, nsample, 3]
        如果returnfps为True，还会返回grouped_xyz和fps_idx
    """
    B, N, C = xyz.shape  # 获取批次大小、点数量和坐标维度
    S = npoint  # 采样点数量
    
    # 步骤1：使用最远点采样选择中心点
    fps_idx = farthest_point_sample(xyz, npoint)  # 获取采样点的索引，形状为[B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # 获取采样点的坐标，形状为[B, npoint, 3]
    
    # 步骤2：对每个中心点进行球查询，找到邻域内的点
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # 获取每个中心点邻域内的点的索引
    grouped_xyz = index_points(xyz, idx)  # 获取邻域点的坐标，形状为[B, npoint, nsample, 3]
    
    # 步骤3：计算邻域点相对于中心点的坐标偏移
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 中心化坐标

    # 步骤4：组合坐标偏移和原始特征（如果有）
    if points is not None:
        grouped_points = index_points(points, idx)  # 获取邻域点的特征
        # 将坐标偏移和特征拼接，形成新的特征
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm  # 如果没有额外特征，则只使用坐标偏移
    
    # 根据returnfps参数决定返回值
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """全局分组函数，将所有点作为一个组处理
    
    这个函数用于PointNet++中的全局特征提取，不进行采样，而是将所有点视为一个组
    
    参数:
        xyz: 输入点的位置数据，形状为[B, N, 3]，其中B为批次大小，N为点数量
        points: 输入点的特征数据，形状为[B, N, D]，其中D为特征维度
    返回:
        new_xyz: 采样后的点位置数据（实际上是原点），形状为[B, 1, 3]
        new_points: 分组后的点特征数据，形状为[B, 1, N, 3+D]或[B, 1, N, 3]
    """
    device = xyz.device  # 获取设备信息
    B, N, C = xyz.shape  # 获取批次大小、点数量和坐标维度
    
    # 创建一个虚拟中心点（原点）
    new_xyz = torch.zeros(B, 1, C).to(device)  # 形状为[B, 1, 3]，表示每个批次一个中心点
    
    # 将所有点视为一个组，不进行采样
    grouped_xyz = xyz.view(B, 1, N, C)  # 形状为[B, 1, N, 3]，表示每个批次所有点属于同一组
    
    # 组合坐标和原始特征（如果有）
    if points is not None:
        # 将坐标和特征拼接，形成新的特征
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # [B, 1, N, 3+D]
    else:
        new_points = grouped_xyz  # 如果没有额外特征，则只使用坐标
    
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet++中的集合抽象层（Set Abstraction Layer）
    
    该层实现了点云的下采样、局部区域分组和特征提取
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """初始化集合抽象层
        
        参数:
            npoint: 采样点数量，如果为None则表示不进行采样（用于全局特征）
            radius: 球查询的半径，如果为None则表示不进行球查询（用于全局特征）
            nsample: 每个局部区域内的最大采样点数，如果为None则表示不进行采样（用于全局特征）
            in_channel: 输入特征的通道数
            mlp: 多层感知机的通道配置列表，如[64, 64, 128]表示三层MLP，输出通道分别为64, 64, 128
            group_all: 是否将所有点作为一个组处理，用于全局特征提取
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 采样点数量
        self.radius = radius  # 球查询半径
        self.nsample = nsample  # 每个局部区域的采样点数
        
        # 创建MLP的卷积层和批归一化层
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        # 构建MLP网络
        last_channel = in_channel  # 初始输入通道数
        for out_channel in mlp:
            # 使用1x1卷积实现MLP
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel  # 更新通道数
            
        self.group_all = group_all  # 是否将所有点作为一个组处理

    def forward(self, xyz, points):
        """前向传播函数
        
        参数:
            xyz: 输入点的位置数据，形状为[B, C, N]，其中B为批次大小，C为坐标维度（通常为3），N为点数量
            points: 输入点的特征数据，形状为[B, D, N]，其中D为特征维度，如果为None则只使用坐标信息
        返回:
            new_xyz: 采样后的点位置数据，形状为[B, C, S]，其中S为采样点数量
            new_points: 提取的特征数据，形状为[B, D', S]，其中D'为特征维度（由MLP的最后一层决定）
        """
        # 调整维度顺序以适应后续处理
        xyz = xyz.permute(0, 2, 1)  # 从[B, C, N]变为[B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # 从[B, D, N]变为[B, N, D]

        # 根据group_all参数选择不同的采样和分组方式
        if self.group_all:
            # 全局特征提取，不进行采样，将所有点作为一个组
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # 局部特征提取，使用最远点采样和球查询
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            
        # new_xyz: 采样点位置，形状为[B, npoint, C]
        # new_points: 采样点及其邻域的特征，形状为[B, npoint, nsample, C+D]或[B, npoint, nsample, C]
        
        # 调整维度顺序以适应卷积操作
        new_points = new_points.permute(0, 3, 2, 1)  # 变为[B, C+D, nsample, npoint]
        
        # 通过MLP处理特征
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))  # 卷积+批归一化+ReLU

        # 使用最大池化聚合每个局部区域的特征
        new_points = torch.max(new_points, 2)[0]  # 在nsample维度上进行最大池化，得到[B, D', npoint]
        
        # 调整输出的维度顺序
        new_xyz = new_xyz.permute(0, 2, 1)  # 从[B, npoint, C]变为[B, C, npoint]
        
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet++中的多尺度集合抽象层（Multi-Scale Grouping Set Abstraction Layer）
    
    该层实现了点云的下采样和多尺度特征提取，通过使用不同半径的球查询来捕获不同尺度的局部特征
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """初始化多尺度集合抽象层
        
        参数:
            npoint: 采样点数量
            radius_list: 不同尺度的球查询半径列表，如[0.1, 0.2, 0.4]
            nsample_list: 对应每个半径的采样点数列表，如[16, 32, 128]
            in_channel: 输入特征的通道数
            mlp_list: 每个尺度对应的MLP配置列表，如[[32,32], [64,64], [64,96]]
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint  # 采样点数量
        self.radius_list = radius_list  # 不同尺度的球查询半径列表
        self.nsample_list = nsample_list  # 对应每个半径的采样点数列表
        
        # 创建多个尺度的卷积块和批归一化块
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        # 为每个尺度创建对应的MLP网络
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()  # 当前尺度的卷积层列表
            bns = nn.ModuleList()    # 当前尺度的批归一化层列表
            
            # 输入通道数为原始特征通道数加上坐标维度(3)
            last_channel = in_channel + 3
            
            # 构建当前尺度的MLP网络
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))  # 1x1卷积
                bns.append(nn.BatchNorm2d(out_channel))  # 批归一化
                last_channel = out_channel  # 更新通道数
                
            # 将当前尺度的网络添加到模块列表中
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """前向传播函数
        
        参数:
            xyz: 输入点的位置数据，形状为[B, C, N]，其中B为批次大小，C为坐标维度（通常为3），N为点数量
            points: 输入点的特征数据，形状为[B, D, N]，其中D为特征维度，如果为None则只使用坐标信息
        返回:
            new_xyz: 采样后的点位置数据，形状为[B, C, S]，其中S为采样点数量
            new_points_concat: 多尺度特征连接后的特征数据，形状为[B, D', S]，其中D'为所有尺度特征维度的总和
        """
        # 调整维度顺序以适应后续处理
        xyz = xyz.permute(0, 2, 1)  # 从[B, C, N]变为[B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # 从[B, D, N]变为[B, N, D]

        B, N, C = xyz.shape  # B:批次大小，N:点数量，C:坐标维度
        S = self.npoint  # 采样点数量
        
        # 使用最远点采样选择中心点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # [B, S, C]
        
        # 存储不同尺度的特征
        new_points_list = []
        
        # 对每个尺度进行处理
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]  # 当前尺度的采样点数
            
            # 球查询找到邻域点
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # [B, S, K]
            
            # 获取邻域点的坐标
            grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, C]
            
            # 计算相对坐标（中心化）
            grouped_xyz -= new_xyz.view(B, S, 1, C)  # [B, S, K, C]
            
            # 合并坐标和特征
            if points is not None:
                # 获取邻域点的特征
                grouped_points = index_points(points, group_idx)  # [B, S, K, D]
                # 将特征和相对坐标拼接
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, S, K, D+C]
            else:
                # 如果没有额外特征，则只使用相对坐标作为特征
                grouped_points = grouped_xyz  # [B, S, K, C]

            # 调整维度顺序以适应卷积操作
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+C, K, S]
            
            # 通过当前尺度的MLP处理特征
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))  # 卷积+批归一化+ReLU
                
            # 使用最大池化聚合每个局部区域的特征
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            
            # 将当前尺度的特征添加到列表中
            new_points_list.append(new_points)

        # 调整输出的维度顺序
        new_xyz = new_xyz.permute(0, 2, 1)  # 从[B, S, C]变为[B, C, S]
        
        # 将不同尺度的特征在通道维度上连接
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, ∑D', S]
        
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """PointNet++中的特征传播层（Feature Propagation Layer）
    
    该层实现了从下采样点到原始点的特征传播（上采样），用于分割任务中的特征恢复
    """
    def __init__(self, in_channel, mlp):
        """初始化特征传播层
        
        参数:
            in_channel: 输入特征的通道数（包括插值特征和跳跃连接特征的通道总和）
            mlp: 多层感知机的通道配置列表，如[128, 128, 64]
        """
        super(PointNetFeaturePropagation, self).__init__()
        
        # 创建MLP的卷积层和批归一化层
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        # 构建MLP网络
        last_channel = in_channel  # 初始输入通道数
        for out_channel in mlp:
            # 使用1x1卷积实现MLP（这里使用Conv1d而不是Conv2d，因为处理的是点特征而不是局部区域特征）
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel  # 更新通道数

    def forward(self, xyz1, xyz2, points1, points2):
        """前向传播函数
        
        参数:
            xyz1: 目标点的位置数据（需要获得特征的点），形状为[B, C, N]，其中N为点数量
            xyz2: 源点的位置数据（已有特征的点），形状为[B, C, S]，其中S为采样点数量
            points1: 目标点的特征数据（如果有的话，用于跳跃连接），形状为[B, D, N]
            points2: 源点的特征数据，形状为[B, D', S]
        返回:
            new_points: 上采样后的特征数据，形状为[B, D'', N]，其中D''为MLP的最后一层输出通道数
        """
        # 调整维度顺序以适应后续处理
        xyz1 = xyz1.permute(0, 2, 1)  # 从[B, C, N]变为[B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # 从[B, C, S]变为[B, S, C]

        points2 = points2.permute(0, 2, 1)  # 从[B, D', S]变为[B, S, D']
        B, N, C = xyz1.shape  # B:批次大小，N:目标点数量，C:坐标维度
        _, S, _ = xyz2.shape  # S:源点数量

        # 特征插值：从源点(S)到目标点(N)
        if S == 1:
            # 如果只有一个源点，则直接复制其特征到所有目标点
            interpolated_points = points2.repeat(1, N, 1)  # [B, N, D']
        else:
            # 计算目标点到源点的距离矩阵
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            
            # 对每个目标点，按距离排序找到最近的3个源点
            dists, idx = dists.sort(dim=-1)  # [B, N, S]
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            # 计算基于距离的权重（距离越近权重越大）
            dist_recip = 1.0 / (dists + 1e-8)  # 避免除零
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 归一化因子
            weight = dist_recip / norm  # [B, N, 3]
            
            # 使用权重对最近的3个源点的特征进行加权平均
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, D']

        # 如果有跳跃连接特征，则与插值特征拼接
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # 从[B, D, N]变为[B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D+D']
        else:
            new_points = interpolated_points  # [B, N, D']

        # 调整维度顺序以适应卷积操作
        new_points = new_points.permute(0, 2, 1)  # 从[B, N, D+D']变为[B, D+D', N]
        
        # 通过MLP处理特征
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))  # 卷积+批归一化+ReLU
            
        return new_points  # [B, D'', N]

