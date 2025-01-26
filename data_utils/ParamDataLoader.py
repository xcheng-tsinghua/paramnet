# -*- coding: gbk -*-
"""
任何数据集加载类
返回索引为 0 的数据为点
返回索引为 1 的数据为类别
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import shutil
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from tqdm import tqdm


class ParamDataLoader(Dataset):
    '''参数化带约束草图数据集，parametric PointNet dataset
    返回：
    ① 点
    ② 类型
    ③ 约束矩阵
    读取数据：前三列为xyz，最后一列为约束
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew', # 数据集文件夹路径
                 npoints=2500, # 每个点云文件的点数
                 is_train=True, # 判断返回训练数据集还是测试集
                 data_augmentation=True, # 是否加噪音
                 is_backaddattr=True
                 ):
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_backaddattr = is_backaddattr

        if(is_train):
            file_ind = os.path.join(root, 'train_files.txt')
        else:
            file_ind = os.path.join(root, 'test_files.txt')

        print('index file path: ', file_ind)

        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        with open(file_ind, 'r', encoding="utf-8") as f:
            for line in f:
                current_line = line.strip().split(',')
                category_path[current_line[0]] = [os.path.join(root, current_line[0], ind_str + '.txt') for ind_str in current_line[1:]]  # 创建每类的键及对应的值的类型，字典的键为字符串 ‘plane','car'

        self.datapath = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        for item in category_path:  # item 为字典的键，即类型‘plane','car'
            for fn in category_path[item]:  # fn 为每类点云对应的文件路径
                self.datapath.append((item, fn)) # item：类型（‘plane','car'）

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # 用整形0,1,2,3等代表具体类型‘plane','car'等，此时字典category_path中的键值没有用到，self.classes的键为‘plane'或'car'，值为0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        fn = self.datapath[index]  # (‘plane’, Path1). fn:一维元组，fn[0]：‘plane'或者'car'，fn[1]：对应的点云文件路径
        cls = self.classes[fn[0]]  # 表示类别的整形数字。 self.classes：键：‘plane'或者'car'，值：用于表示其类型的整形数字 0,1

        # point_set = np.loadtxt(fn[1], delimiter=',',dtype=np.float32)  # n*8 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------
        point_set = np.loadtxt(fn[1])  # n*8 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        if self.is_backaddattr:
            # 欧拉角的标签
            eualangle = point_set[:, 3: 6]

            # 是否是边缘点
            is_nearby = point_set[:, 6]

            # 所在基元类型
            meta_type = point_set[:, 7]

        # 点坐标
        point_set = point_set[:, :3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation # 仅仅是x，y分量作旋转-----------
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        # point_set = torch.from_numpy(point_set)
        # cls = np.array([cls]).astype(np.int32)

        if self.is_backaddattr:
            return point_set, cls, eualangle, is_nearby, meta_type
        else:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


class PrismCuboidDataLoader(Dataset):
    '''
    棱柱和长方体点云数据集
    返回：
    ① 点
    ② 类型
    ③ 约束矩阵
    读取数据：前三列为xyz，最后一列为约束
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew', # 数据集文件夹路径
                 npoints=2500, # 每个点云文件的点数
                 is_train=True, # 判断返回训练数据集还是测试集
                 data_augmentation=False, # 是否加噪音
                 prism_angle=50, # [50, 60, 70, 80, 82, 85, 87, 89]
                 instance_all=5000  # 单个类别的点云总数（训练集+测试集）
                 ):
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.instance_all = instance_all
        self.root = root
        self.prism_angle = prism_angle

        if is_train:
            file_ind = os.path.join(root, 'train_files.txt')
        else:
            file_ind = os.path.join(root, 'test_files.txt')

        print('index file path: ', file_ind)

        self.classes = [0, 1]
        self.datapath = []
        with open(file_ind, 'r', encoding="utf-8") as f:
            for line in f:
                current_line = line.strip()
                idx_from_line = int(current_line)

                cls, file_root = self.get_cls_filepth(idx_from_line)
                self.datapath.append((cls, file_root))

        print('instance all:', len(self.datapath))

    def get_cls_filepth(self, idx_from_idx_file: int):
        """
        从索引文件的序号推出对应文件
        其中 0 <= idx < instance_all 的为 cuboid，文件名为 PointCloud{idx}.txt
        其中 instance_all <= idx < 2 * instance_all 的为 prism，文件名为 PointCloud{idx-instance_all}.txt
        点云的类别：cuboid 用整形数字0表示，prism 用整形数字1表示
        """
        if idx_from_idx_file < self.instance_all:
            cls = 0
            file_root = os.path.join(self.root, 'cuboid', f'PointCloud{idx_from_idx_file}.txt')
        else:
            cls = 1
            file_root = os.path.join(self.root, f'prism{self.prism_angle}', f'PointCloud{idx_from_idx_file - self.instance_all}.txt')

        return cls, file_root

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        fn = self.datapath[index]  # (‘plane’, Path1). fn:一维元组，fn[0]：‘plane'或者'car'，fn[1]：对应的点云文件路径
        cls = fn[0]  # 表示类别的整形数字。 self.classes：键：‘plane'或者'car'，值：用于表示其类型的整形数字 0,1

        point_set = np.loadtxt(fn[1])  # n*4 的矩阵,存储点数据,前三列为xyz,最后一列为约束c,------------------

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        # 点坐标
        xyz = point_set[:, :3]
        cst = point_set[:, 3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # xyz[:, [0, 2]] = xyz[:, [0, 2]].dot(rotation_matrix)  # random rotation # 仅仅是x，y分量作旋转-----------
            xyz += np.random.normal(0, 0.02, size=xyz.shape)  # random jitter # 所有分量加正态分布随机数

        return xyz, cls, cst

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return self.classes


class STEPMillionDataLoader(Dataset):
    '''参数化带约束草图数据集，parametric PointNet dataset
    返回：
    ① 点
    ② 类型
    ③ 约束矩阵
    读取数据：前三列为xyz，最后一列为约束
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew',  # 数据集文件夹路径
                 npoints=2500,  # 每个点云文件的点数
                 data_augmentation=True,  # 是否加噪音
                 is_backaddattr=True
                 ):

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_backaddattr = is_backaddattr

        print('STEPMillion dataset, from:' + root)

        self.datapath = get_allfiles(root)


        # index_file = os.path.join(root, 'index_file.txt')
        #
        # self.datapath = []
        # with open(index_file, 'r', encoding="utf-8") as f:
        #     for line in f.readlines():
        #         current_path = os.path.join(root, 'overall', line)
        #         self.datapath.append(current_path)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        # 找到对应文件路径
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, near, meta]

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn, '------', fn)
            exit('except an error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        if self.is_backaddattr:
            # 欧拉角的标签
            eualangle = point_set[:, 3: 6]

            # 是否是边缘点
            is_nearby = point_set[:, 6]

            # 所在基元类型
            meta_type = point_set[:, 7]

        # 点坐标
        point_set = point_set[:, :3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation # 仅仅是x，y分量作旋转-----------
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        if self.is_backaddattr:
            return point_set, eualangle, is_nearby, meta_type
        else:
            return point_set

    def __len__(self):
        return len(self.datapath)


class MCBDataLoader(Dataset):

    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A',  # 数据集文件夹路径
                 is_train=True,
                 npoints=2500,  # 每个点云文件的点数
                 data_augmentation=True,  # 是否加噪音
                 is_back_addattr=False
                 ):
        """
        定位文件的路径如下：
        root
        ├─ train
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │
        ├─ test
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │

        """

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_back_addattr = is_back_addattr

        print('MCB dataset, from:' + root)

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # 获取全部类别列表，即 inner_root 内的全部文件夹名
        category_all = get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        for item in category_path:  # item 为字典的键，即类型‘plane','car'
            for fn in category_path[item]:  # fn 为每类点云对应的文件路径
                self.datapath.append((item, fn)) # item：类型（‘plane','car'）

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # 用整形0,1,2,3等代表具体类型‘plane','car'等，此时字典category_path中的键值没有用到，self.classes的键为‘plane'或'car'，值为0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        fn = self.datapath[index]  # (‘plane’, Path1). fn:一维元组，fn[0]：‘plane'或者'car'，fn[1]：对应的点云文件路径
        cls = self.classes[fn[0]]  # 表示类别的整形数字。 self.classes：键：‘plane'或者'car'，值：用于表示其类型的整形数字 0,1
        point_set = np.loadtxt(fn[1])  # n*6 (x, y, z, i, j, k)

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        if self.is_back_addattr:
            euler = point_set[:, 3:6]
            near = point_set[:, 6]
            meta = point_set[:, 7]

        # 点坐标
        point_set = point_set[:, :3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        if self.is_back_addattr:
            return point_set, cls, euler, near, meta

        else:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


class Seg360GalleryDataLoader(Dataset):
    """
    读取 360 Gallery 分割数据集
    360 Gallery 数据集中的每个点云文件点数为 2048
    """
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A',  # 数据集文件夹路径
                 is_train=True,
                 npoints=2000,  # 每个点云文件的采样点数
                 data_augmentation=False,  # 是否加噪音
                 ):
        """
        定位文件的路径如下：
        root
        ├─ point_clouds
        │   ├─ 0.findx        root
        ├─ point_clouds
        │   ├─ 0.findx
        │   ├─ 0.seg
        │   ├─ 0.xyz
        │   │
        │   ├─ 1.findx
        │   ├─ 1.seg
        │   ├─ 1.xyz
        │   │
        │   ├─ 2.findx
        │   ├─ 2.seg
        │   ├─ 2.xyz
        │   │
        │   ...
        │
        ├─ segment_names.json
        └─ train_test.json
        │   ├─ 0.seg
        │   ├─ 0.xyz
        │   │
        │   ├─ 1.findx
        │   ├─ 1.seg
        │   ├─ 1.xyz
        │   │
        │   ├─ 2.findx
        │   ├─ 2.seg
        │   ├─ 2.xyz
        │   │
        │   ...
        │
        ├─ segment_names.json
        └─ train_test.json

        """
        print('360Gallery Segmentation dataset, from:' + root)

        self.npoints = npoints
        self.data_augmentation = data_augmentation

        with open(os.path.join(root, 'train_test.json'), 'r') as file_json:
            train_test_filename = json.load(file_json)

        with open(os.path.join(root, 'segment_names.json'), 'r') as file_json:
            self.seg_names = json.load(file_json)

        if is_train:
            file_names = train_test_filename["train"]
        else:
            file_names = train_test_filename["test"]

        self.datapath = []  # [(xyz_filename, seg_filename), ...]存储点云的绝对路径及分割文件的绝对路径

        for c_file_name in file_names:
            xyz_file_path = os.path.join(root, 'point_clouds', c_file_name + '.xyz')
            seg_file_path = os.path.join(root, 'point_clouds', c_file_name + '.seg')

            self.datapath.append((xyz_file_path, seg_file_path))

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[0])  # n*6 (x, y, z, i, j, k)
        seg_label = np.loadtxt(fn[1])

        point_and_seg = np.concatenate((point_set[:, :3], seg_label.reshape(-1, 1)), axis=1)

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_and_seg.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_and_seg = point_and_seg[choice, :]

        # 点坐标
        point_set = point_and_seg[:, :3]

        # 分割标签
        seg_label = point_and_seg[:, -1]

        # 先减去质心，再除最大距离，即进行位置和大小的归一化
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        return point_set, seg_label

    def __len__(self):
        return len(self.datapath)


class STEP9000DataLoader(Dataset):
    def __init__(self,
                 root,  # 数据集文件夹路径
                 is_train=True,
                 npoints=2000,  # 每个点云文件的点数
                 data_augmentation=True,  # 是否加噪音
                 ):
        """
        定位文件的路径如下：
        root
        ├─ train
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │
        ├─ test
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │

        """

        self.npoints = npoints
        self.data_augmentation = data_augmentation

        print('STEP9000 dataset, from:' + root)

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # 获取全部类别列表，即 inner_root 内的全部文件夹名
        category_all = get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []  # [(‘plane’, Path1), (‘car’, Path1), ...]存储点云的绝对路径，此外还有类型，放入同一个数组。类型，点云构成数组中的一个元素
        for item in category_path:  # item 为字典的键，即类型‘plane','car'
            for fn in category_path[item]:  # fn 为每类点云对应的文件路径
                self.datapath.append((item, fn)) # item：类型（‘plane','car'）

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # 用整形0,1,2,3等代表具体类型‘plane','car'等，此时字典category_path中的键值没有用到，self.classes的键为‘plane'或'car'，值为0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # 作用为输出点云中点的三维坐标及对应类别，类型均为tensor，类别为1×1的矩阵
        fn = self.datapath[index]  # (‘plane’, Path1). fn:一维元组，fn[0]：‘plane'或者'car'，fn[1]：对应的点云文件路径
        cls = self.classes[fn[0]]  # 表示类别的整形数字。 self.classes：键：‘plane'或者'car'，值：用于表示其类型的整形数字 0,1
        point_set = np.loadtxt(fn[1])  # n*6 (x, y, z, i, j, k)

        # 从 np.arange(len(seg))中随机选数，数量为self.npoints，replace：是否可取相同数字，replace=true表示可取相同数字，可规定每个元素的抽取概率，默认均等概率
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('捕获异常时的point_set.shape[0]：', point_set.shape[0])
            print('捕获出错时的文件：', fn[0], '------', fn[1])
            exit('except a error')

        # 从读取到的点云文件中随机取指定数量的点
        point_set = point_set[choice, :]

        # 欧拉角
        euler_angle = point_set[:, 3:6]

        # 邻近边
        edge_nearby = point_set[:, 6]

        # 基元类型
        meta_type = point_set[:, 7]

        # 点坐标
        point_set = point_set[:, :3]

        # 先减去平均点，再除最大距离，即进行位置和大小的归一化
        # center # np.mean() 返回一个1*x的矩阵，axis=0 求每列的均值，axis=1，求每行的均值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 随机旋转及加上正态分布噪音
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # 所有分量加正态分布随机数

        return point_set, cls, euler_angle, edge_nearby, meta_type

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


class ShapeNet50SegDataLoader(Dataset):
    def __init__(self,
                 root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 npoints=2500,
                 split='train',
                 class_choice=None,
                 normal_channel=False
                 ):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {
            'Airplane': [0, 1, 2, 3],
            'Bag': [4, 5],
            'Cap': [6, 7],
            'Car': [8, 9, 10, 11],
            'Chair': [12, 13, 14, 15],
            'Earphone': [16, 17, 18],
            'Guitar': [19, 20, 21],
            'Knife': [22, 23],
            'Lamp': [24, 25, 26, 27],
            'Laptop': [28, 29],
            'Motorbike': [30, 31, 32, 33, 34, 35],
            'Mug': [36, 37],
            'Pistol': [38, 39, 40],
            'Rocket': [41, 42, 43],
            'Skateboard': [44, 45, 46],
            'Table': [47, 48, 49]
        }

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        '''
        返回
        [0]点集 (2048, 3) <class 'numpy.ndarray'>
        [1]类别 (1,) <class 'numpy.ndarray'>
        [2]分割 (2048,) <class 'numpy.ndarray'>
        '''
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.split('.')[-1] == suffix:
                if(filename_only):
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


def vis_confusion_mat(file_name):
    '''
    第一行 predict，第二行 target
    :param file_name:
    :return:
    '''
    array_from_file = np.loadtxt(file_name, dtype=int)

    # 确定矩阵的大小（假设最大值为5，因此矩阵大小为6x6）
    matrix_size = array_from_file.max() + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # 遍历 list1 和 list2 并更新矩阵
    for i in range(array_from_file.shape[1]):
        x = array_from_file[0, i]
        y = array_from_file[1, i]
        matrix[x, y] += 1

    # 打印矩阵以查看结果
    print("矩阵：")
    print(matrix)

    # 使用 Matplotlib 可视化矩阵
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.show()


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    # 确定矩阵的大小（假设最大值为5，因此矩阵大小为6x6）
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    # 遍历 list1 和 list2 并更新矩阵
    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    # 使用 Matplotlib 可视化矩阵
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.savefig(save_name)
    plt.close()


def save_dir2gif(dir_path, gif_path='output.gif'):
    # 获取所有图片文件路径
    images = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(('png', 'jpg', 'jpeg'))]

    # 打开图片并存储在一个列表中
    frames = [Image.open(image) for image in images]

    # 将图片保存为GIF
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)


def search_fileall(dir_path):
    '''
    获取某个文件夹下全部文件路径
    :param dir_path: 目标文件夹
    :return: 全部路径数组
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            filepath_all.append(os.path.join(root, file))

    return filepath_all


def metatype_statistic():
    # 先找到所有文件
    dirpath = r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud'
    all_files = search_fileall(dirpath)[2:]

    all_metatype = []
    for filepath in all_files:
        point_set = np.loadtxt(filepath)
        all_metatype.append(point_set[:, 7])

    all_metatype = np.concatenate(all_metatype)
    # 使用unique函数获取唯一的元素及其计数
    unique_elements, counts = np.unique(all_metatype, return_counts=True)

    # 输出统计结果
    for element, count in zip(unique_elements, counts):
        print(f"数字 {element} 出现了 {count} 次")


def segfig_save(points, seg_pred, save_path):
    plt.axis('off')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    points_show = points[0].cpu().numpy().copy()
    ax.clear()

    # Hide the background grid
    ax.grid(False)
    # Hide the axes
    ax.set_axis_off()
    # Alternatively, you can hide only the ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    x_data = points_show[:, 0]
    y_data = points_show[:, 1]
    z_data = points_show[:, 2]
    c_data = seg_pred[0].max(dim=1)[1].cpu().numpy()

    ax.scatter(x_data, y_data, z_data, c=c_data, s=100, edgecolors='none')
    plt.savefig(save_path)
    plt.close()


def gallery360seg_count():
    # 找到全部分割文件
    dir_path = r'D:\document\DeepLearning\DataSet\360Gallery_Seg\point_clouds'
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.split('.')[-1] == 'seg':
                current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    count_dict = {}
    for i in range(8):
        count_dict[i] = 0

    c_files_seem = ''

    for c_file in tqdm(filepath_all, total = len(filepath_all)):

        with open(c_file, 'r') as file:
            # 读取文件中的每一行，?并去除行尾的换行符
            numbers = [int(line.strip()) for line in file]

            if 7 in numbers:
                print('存在7的文件：', c_file)

            for number in numbers:
                if number in count_dict:
                    count_dict[number] += 1
                else:
                    print('error key occur:', number)

        is_all_searched = True
        for i in range(8):
            is_all_searched = is_all_searched and count_dict[i]

        if is_all_searched:
            filename_with_extension = os.path.basename(c_file)
            filename_without_extension = os.path.splitext(filename_with_extension)[0]
            c_files_seem += '\"' + filename_without_extension + '\"' + ',\n'
            print(c_files_seem)
            print(count_dict)
            exit(0)
        else:
            filename_with_extension = os.path.basename(c_file)
            filename_without_extension = os.path.splitext(filename_with_extension)[0]
            c_files_seem += '\"' + filename_without_extension + '\"' + ',\n'

    print(count_dict)


def gallery360seg_test():
    with open(r'D:\document\DeepLearning\DataSet\360Gallery_Seg\train_test.json', 'r') as file_json:
        train_test_filename = json.load(file_json)
    file_names = train_test_filename["train"]

    count_dict = {}
    for i in range(8):
        count_dict[i] = 0

    for c_file in tqdm(file_names, total = len(file_names)):
        c_file_path = os.path.join(r'D:\document\DeepLearning\DataSet\360Gallery_Seg\point_clouds', c_file + '.seg')
        with open(c_file_path, 'r') as file:
            # 读取文件中的每一行，?并去除行尾的换行符
            numbers = [int(line.strip()) for line in file]

            for number in numbers:
                count_dict[number] += 1
                # if number == 7:
                #     print('发现7', c_file_path)
                #     exit(0)

    print(count_dict)





if __name__ == '__main__':
    adata = STEP9000DataLoader(r'D:\document\DeepLearning\DataSet\STEP9000\STEP9000_test\pcd', is_train=True)
    adatatest = STEP9000DataLoader(r'D:\document\DeepLearning\DataSet\STEP9000\STEP9000_test\pcd', is_train=False)

    lentrain = len(adata)
    atrain = adata[lentrain - 1]

    lentest = len(adatatest)
    atest = adata[lentest - 1]

    aasa = 0

    # vis_confusion_mat('./confusion/cf_mat0.txt')
    # gallery360seg_count()
    # gallery360seg_test()
    # print(33567527 + 26694774+ 2681385+ 838830+1957997+469522+6771395+91210 - 35680 * 2048)

    # save_dir2gif(r'C:\Users\ChengXi\Desktop\CA_two_path-2024-06-03 08-24-49\train')

    # index_file()
    # file_copy()
    # disp_pointcloud()
    # test_toOneHot()
    # test_is_normal_normalized()
    # metatype_statistic()
    # search_STEPall(r'D:\document\DeepLearning\DataSet\STEPMillion\raw')
    # generate_class_path_file(r'D:\document\DeepLearning\DataSet\900\raw')

    # prepare_for_pointcloud_generate(r'D:\document\DeepLearning\DataSet\900')
    # index_file(r'D:\document\DeepLearning\DataSet\900\pointcloud')


    # all_files = search_fileall(r'D:\document\DeepLearning\DataSet\900\filepath')
    # with open('filepath.txt', "w") as filewrite:
    #     for afile in all_files:
    #         filewrite.write(afile + '\n')


    pass
