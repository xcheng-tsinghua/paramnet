import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    # print('distance.shape: ', distance.shape)

    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    return neighbor_direction_norm


class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""

    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)  # (3, s * k)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature


class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support,
                                            neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature_fuse


class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


def farthest_point_sample(xyz, npoint):
    """
    最远采样法进行采样，返回采样点的索引
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    # 生成 B 行，npoint 列的全为零的矩阵
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)

    # 生成 B 行，N 列的矩阵，每个元素为 1e10
    distance = torch.ones(B, N).to(device) * 1e10

    # 生成随机整数tensor，整数范围在[0，N)之间，包含0不包含N，矩阵各维度长度必须用元组传入，因此写成(B,)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    返回 points 中 索引 idx 对应的点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PoolTo_layer(nn.Module):
    '''
    指定池化到的点数
    点使用farthest point sampling 选取
    '''

    def __init__(self, pooling_to: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_to = pooling_to
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        # 聚集特征
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        # 缩小点规模
        sampled_idx = farthest_point_sample(vertices, self.pooling_to)
        vertices_pool = index_points(vertices, sampled_idx)
        feature_map_pool = index_points(pooled_feature, sampled_idx)

        # pool_num = self.pooling_to
        # sample_idx = torch.randperm(vertice_num)[:pool_num]
        # vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        # feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


class GCN3D(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int, classes_num: int =40):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = Conv_layer(256, 1024, support_num= support_num)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Linear(256, classes_num)
        )

    def forward(self,  vertices: "(bs, 3, vertice_num)"):
        vertices = vertices.permute(0, 2, 1)
        bs, vertice_num, _ = vertices.size()

        # 计算最近的 self.neighbor_num 个点的索引
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        # neighbor_index->(bs, vertice_num, neighbor_num)

        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True)
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        feature_global = fm_4.max(1)[0]
        pred = self.classifier(feature_global)

        pred = F.log_softmax(pred, dim=1)
        return pred

