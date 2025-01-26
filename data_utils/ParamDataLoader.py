# -*- coding: gbk -*-
"""
�κ����ݼ�������
��������Ϊ 0 ������Ϊ��
��������Ϊ 1 ������Ϊ���
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
    '''��������Լ����ͼ���ݼ���parametric PointNet dataset
    ���أ�
    �� ��
    �� ����
    �� Լ������
    ��ȡ���ݣ�ǰ����Ϊxyz�����һ��ΪԼ��
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew', # ���ݼ��ļ���·��
                 npoints=2500, # ÿ�������ļ��ĵ���
                 is_train=True, # �жϷ���ѵ�����ݼ����ǲ��Լ�
                 data_augmentation=True, # �Ƿ������
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
                category_path[current_line[0]] = [os.path.join(root, current_line[0], ind_str + '.txt') for ind_str in current_line[1:]]  # ����ÿ��ļ�����Ӧ��ֵ�����ͣ��ֵ�ļ�Ϊ�ַ��� ��plane','car'

        self.datapath = []  # [(��plane��, Path1), (��car��, Path1), ...]�洢���Ƶľ���·�������⻹�����ͣ�����ͬһ�����顣���ͣ����ƹ��������е�һ��Ԫ��
        for item in category_path:  # item Ϊ�ֵ�ļ��������͡�plane','car'
            for fn in category_path[item]:  # fn Ϊÿ����ƶ�Ӧ���ļ�·��
                self.datapath.append((item, fn)) # item�����ͣ���plane','car'��

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # ������0,1,2,3�ȴ���������͡�plane','car'�ȣ���ʱ�ֵ�category_path�еļ�ֵû���õ���self.classes�ļ�Ϊ��plane'��'car'��ֵΪ0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # ����Ϊ��������е����ά���꼰��Ӧ������;�Ϊtensor�����Ϊ1��1�ľ���
        fn = self.datapath[index]  # (��plane��, Path1). fn:һάԪ�飬fn[0]����plane'����'car'��fn[1]����Ӧ�ĵ����ļ�·��
        cls = self.classes[fn[0]]  # ��ʾ�����������֡� self.classes��������plane'����'car'��ֵ�����ڱ�ʾ�����͵��������� 0,1

        # point_set = np.loadtxt(fn[1], delimiter=',',dtype=np.float32)  # n*8 �ľ���,�洢������,ǰ����Ϊxyz,���һ��ΪԼ��c,------------------
        point_set = np.loadtxt(fn[1])  # n*8 �ľ���,�洢������,ǰ����Ϊxyz,���һ��ΪԼ��c,------------------

        # �� np.arange(len(seg))�����ѡ��������Ϊself.npoints��replace���Ƿ��ȡ��ͬ���֣�replace=true��ʾ��ȡ��ͬ���֣��ɹ涨ÿ��Ԫ�صĳ�ȡ���ʣ�Ĭ�Ͼ��ȸ���
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn[0], '------', fn[1])
            exit('except a error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_set = point_set[choice, :]

        if self.is_backaddattr:
            # ŷ���ǵı�ǩ
            eualangle = point_set[:, 3: 6]

            # �Ƿ��Ǳ�Ե��
            is_nearby = point_set[:, 6]

            # ���ڻ�Ԫ����
            meta_type = point_set[:, 7]

        # ������
        point_set = point_set[:, :3]

        # �ȼ�ȥƽ���㣬�ٳ������룬������λ�úʹ�С�Ĺ�һ��
        # center # np.mean() ����һ��1*x�ľ���axis=0 ��ÿ�еľ�ֵ��axis=1����ÿ�еľ�ֵ
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation # ������x��y��������ת-----------
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # ���з�������̬�ֲ������

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
    �����ͳ�����������ݼ�
    ���أ�
    �� ��
    �� ����
    �� Լ������
    ��ȡ���ݣ�ǰ����Ϊxyz�����һ��ΪԼ��
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew', # ���ݼ��ļ���·��
                 npoints=2500, # ÿ�������ļ��ĵ���
                 is_train=True, # �жϷ���ѵ�����ݼ����ǲ��Լ�
                 data_augmentation=False, # �Ƿ������
                 prism_angle=50, # [50, 60, 70, 80, 82, 85, 87, 89]
                 instance_all=5000  # �������ĵ���������ѵ����+���Լ���
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
        �������ļ�������Ƴ���Ӧ�ļ�
        ���� 0 <= idx < instance_all ��Ϊ cuboid���ļ���Ϊ PointCloud{idx}.txt
        ���� instance_all <= idx < 2 * instance_all ��Ϊ prism���ļ���Ϊ PointCloud{idx-instance_all}.txt
        ���Ƶ����cuboid ����������0��ʾ��prism ����������1��ʾ
        """
        if idx_from_idx_file < self.instance_all:
            cls = 0
            file_root = os.path.join(self.root, 'cuboid', f'PointCloud{idx_from_idx_file}.txt')
        else:
            cls = 1
            file_root = os.path.join(self.root, f'prism{self.prism_angle}', f'PointCloud{idx_from_idx_file - self.instance_all}.txt')

        return cls, file_root

    def __getitem__(self, index):  # ����Ϊ��������е����ά���꼰��Ӧ������;�Ϊtensor�����Ϊ1��1�ľ���
        fn = self.datapath[index]  # (��plane��, Path1). fn:һάԪ�飬fn[0]����plane'����'car'��fn[1]����Ӧ�ĵ����ļ�·��
        cls = fn[0]  # ��ʾ�����������֡� self.classes��������plane'����'car'��ֵ�����ڱ�ʾ�����͵��������� 0,1

        point_set = np.loadtxt(fn[1])  # n*4 �ľ���,�洢������,ǰ����Ϊxyz,���һ��ΪԼ��c,------------------

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn[0], '------', fn[1])
            exit('except a error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_set = point_set[choice, :]

        # ������
        xyz = point_set[:, :3]
        cst = point_set[:, 3]

        # �ȼ�ȥƽ���㣬�ٳ������룬������λ�úʹ�С�Ĺ�һ��
        # center # np.mean() ����һ��1*x�ľ���axis=0 ��ÿ�еľ�ֵ��axis=1����ÿ�еľ�ֵ
        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0) # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # xyz[:, [0, 2]] = xyz[:, [0, 2]].dot(rotation_matrix)  # random rotation # ������x��y��������ת-----------
            xyz += np.random.normal(0, 0.02, size=xyz.shape)  # random jitter # ���з�������̬�ֲ������

        return xyz, cls, cst

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return self.classes


class STEPMillionDataLoader(Dataset):
    '''��������Լ����ͼ���ݼ���parametric PointNet dataset
    ���أ�
    �� ��
    �� ����
    �� Լ������
    ��ȡ���ݣ�ǰ����Ϊxyz�����һ��ΪԼ��
    x y z c
    x y z c
    ...
    x y z c
    '''
    def __init__(self,
                 root=r'D:\document\DeepLearning\ParPartsNetWork\DataSetNew',  # ���ݼ��ļ���·��
                 npoints=2500,  # ÿ�������ļ��ĵ���
                 data_augmentation=True,  # �Ƿ������
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

    def __getitem__(self, index):  # ����Ϊ��������е����ά���꼰��Ӧ������;�Ϊtensor�����Ϊ1��1�ľ���
        # �ҵ���Ӧ�ļ�·��
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, near, meta]

        # �� np.arange(len(seg))�����ѡ��������Ϊself.npoints��replace���Ƿ��ȡ��ͬ���֣�replace=true��ʾ��ȡ��ͬ���֣��ɹ涨ÿ��Ԫ�صĳ�ȡ���ʣ�Ĭ�Ͼ��ȸ���
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn, '------', fn)
            exit('except an error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_set = point_set[choice, :]

        if self.is_backaddattr:
            # ŷ���ǵı�ǩ
            eualangle = point_set[:, 3: 6]

            # �Ƿ��Ǳ�Ե��
            is_nearby = point_set[:, 6]

            # ���ڻ�Ԫ����
            meta_type = point_set[:, 7]

        # ������
        point_set = point_set[:, :3]

        # �ȼ�ȥƽ���㣬�ٳ������룬������λ�úʹ�С�Ĺ�һ��
        # center # np.mean() ����һ��1*x�ľ���axis=0 ��ÿ�еľ�ֵ��axis=1����ÿ�еľ�ֵ
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation # ������x��y��������ת-----------
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # ���з�������̬�ֲ������

        if self.is_backaddattr:
            return point_set, eualangle, is_nearby, meta_type
        else:
            return point_set

    def __len__(self):
        return len(self.datapath)


class MCBDataLoader(Dataset):

    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A',  # ���ݼ��ļ���·��
                 is_train=True,
                 npoints=2500,  # ÿ�������ļ��ĵ���
                 data_augmentation=True,  # �Ƿ������
                 is_back_addattr=False
                 ):
        """
        ��λ�ļ���·�����£�
        root
        ���� train
        ��   ���� Bushes
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ���� Clamps
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ...
        ��
        ���� test
        ��   ���� Bushes
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ���� Clamps
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ...
        ��

        """

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.is_back_addattr = is_back_addattr

        print('MCB dataset, from:' + root)

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # ��ȡȫ������б��� inner_root �ڵ�ȫ���ļ�����
        category_all = get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []  # [(��plane��, Path1), (��car��, Path1), ...]�洢���Ƶľ���·�������⻹�����ͣ�����ͬһ�����顣���ͣ����ƹ��������е�һ��Ԫ��
        for item in category_path:  # item Ϊ�ֵ�ļ��������͡�plane','car'
            for fn in category_path[item]:  # fn Ϊÿ����ƶ�Ӧ���ļ�·��
                self.datapath.append((item, fn)) # item�����ͣ���plane','car'��

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # ������0,1,2,3�ȴ���������͡�plane','car'�ȣ���ʱ�ֵ�category_path�еļ�ֵû���õ���self.classes�ļ�Ϊ��plane'��'car'��ֵΪ0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # ����Ϊ��������е����ά���꼰��Ӧ������;�Ϊtensor�����Ϊ1��1�ľ���
        fn = self.datapath[index]  # (��plane��, Path1). fn:һάԪ�飬fn[0]����plane'����'car'��fn[1]����Ӧ�ĵ����ļ�·��
        cls = self.classes[fn[0]]  # ��ʾ�����������֡� self.classes��������plane'����'car'��ֵ�����ڱ�ʾ�����͵��������� 0,1
        point_set = np.loadtxt(fn[1])  # n*6 (x, y, z, i, j, k)

        # �� np.arange(len(seg))�����ѡ��������Ϊself.npoints��replace���Ƿ��ȡ��ͬ���֣�replace=true��ʾ��ȡ��ͬ���֣��ɹ涨ÿ��Ԫ�صĳ�ȡ���ʣ�Ĭ�Ͼ��ȸ���
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn[0], '------', fn[1])
            exit('except a error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_set = point_set[choice, :]

        if self.is_back_addattr:
            euler = point_set[:, 3:6]
            near = point_set[:, 6]
            meta = point_set[:, 7]

        # ������
        point_set = point_set[:, :3]

        # �ȼ�ȥƽ���㣬�ٳ������룬������λ�úʹ�С�Ĺ�һ��
        # center # np.mean() ����һ��1*x�ľ���axis=0 ��ÿ�еľ�ֵ��axis=1����ÿ�еľ�ֵ
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # ���з�������̬�ֲ������

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
    ��ȡ 360 Gallery �ָ����ݼ�
    360 Gallery ���ݼ��е�ÿ�������ļ�����Ϊ 2048
    """
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A',  # ���ݼ��ļ���·��
                 is_train=True,
                 npoints=2000,  # ÿ�������ļ��Ĳ�������
                 data_augmentation=False,  # �Ƿ������
                 ):
        """
        ��λ�ļ���·�����£�
        root
        ���� point_clouds
        ��   ���� 0.findx        root
        ���� point_clouds
        ��   ���� 0.findx
        ��   ���� 0.seg
        ��   ���� 0.xyz
        ��   ��
        ��   ���� 1.findx
        ��   ���� 1.seg
        ��   ���� 1.xyz
        ��   ��
        ��   ���� 2.findx
        ��   ���� 2.seg
        ��   ���� 2.xyz
        ��   ��
        ��   ...
        ��
        ���� segment_names.json
        ���� train_test.json
        ��   ���� 0.seg
        ��   ���� 0.xyz
        ��   ��
        ��   ���� 1.findx
        ��   ���� 1.seg
        ��   ���� 1.xyz
        ��   ��
        ��   ���� 2.findx
        ��   ���� 2.seg
        ��   ���� 2.xyz
        ��   ��
        ��   ...
        ��
        ���� segment_names.json
        ���� train_test.json

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

        self.datapath = []  # [(xyz_filename, seg_filename), ...]�洢���Ƶľ���·�����ָ��ļ��ľ���·��

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

        # �� np.arange(len(seg))�����ѡ��������Ϊself.npoints��replace���Ƿ��ȡ��ͬ���֣�replace=true��ʾ��ȡ��ͬ���֣��ɹ涨ÿ��Ԫ�صĳ�ȡ���ʣ�Ĭ�Ͼ��ȸ���
        try:
            choice = np.random.choice(point_and_seg.shape[0], self.npoints, replace=False)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn[0], '------', fn[1])
            exit('except a error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_and_seg = point_and_seg[choice, :]

        # ������
        point_set = point_and_seg[:, :3]

        # �ָ��ǩ
        seg_label = point_and_seg[:, -1]

        # �ȼ�ȥ���ģ��ٳ������룬������λ�úʹ�С�Ĺ�һ��
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # ���з�������̬�ֲ������

        return point_set, seg_label

    def __len__(self):
        return len(self.datapath)


class STEP9000DataLoader(Dataset):
    def __init__(self,
                 root,  # ���ݼ��ļ���·��
                 is_train=True,
                 npoints=2000,  # ÿ�������ļ��ĵ���
                 data_augmentation=True,  # �Ƿ������
                 ):
        """
        ��λ�ļ���·�����£�
        root
        ���� train
        ��   ���� Bushes
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ���� Clamps
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ...
        ��
        ���� test
        ��   ���� Bushes
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ���� Clamps
        ��   ��   ����0.obj
        ��   ��   ����1.obj
        ��   ��   ...
        ��   ��
        ��   ...
        ��

        """

        self.npoints = npoints
        self.data_augmentation = data_augmentation

        print('STEP9000 dataset, from:' + root)

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # ��ȡȫ������б��� inner_root �ڵ�ȫ���ļ�����
        category_all = get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = get_allfiles(class_root)

            category_path[c_class] = file_path_all

        self.datapath = []  # [(��plane��, Path1), (��car��, Path1), ...]�洢���Ƶľ���·�������⻹�����ͣ�����ͬһ�����顣���ͣ����ƹ��������е�һ��Ԫ��
        for item in category_path:  # item Ϊ�ֵ�ļ��������͡�plane','car'
            for fn in category_path[item]:  # fn Ϊÿ����ƶ�Ӧ���ļ�·��
                self.datapath.append((item, fn)) # item�����ͣ���plane','car'��

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))  # ������0,1,2,3�ȴ���������͡�plane','car'�ȣ���ʱ�ֵ�category_path�еļ�ֵû���õ���self.classes�ļ�Ϊ��plane'��'car'��ֵΪ0,1
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):  # ����Ϊ��������е����ά���꼰��Ӧ������;�Ϊtensor�����Ϊ1��1�ľ���
        fn = self.datapath[index]  # (��plane��, Path1). fn:һάԪ�飬fn[0]����plane'����'car'��fn[1]����Ӧ�ĵ����ļ�·��
        cls = self.classes[fn[0]]  # ��ʾ�����������֡� self.classes��������plane'����'car'��ֵ�����ڱ�ʾ�����͵��������� 0,1
        point_set = np.loadtxt(fn[1])  # n*6 (x, y, z, i, j, k)

        # �� np.arange(len(seg))�����ѡ��������Ϊself.npoints��replace���Ƿ��ȡ��ͬ���֣�replace=true��ʾ��ȡ��ͬ���֣��ɹ涨ÿ��Ԫ�صĳ�ȡ���ʣ�Ĭ�Ͼ��ȸ���
        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            print('�����쳣ʱ��point_set.shape[0]��', point_set.shape[0])
            print('�������ʱ���ļ���', fn[0], '------', fn[1])
            exit('except a error')

        # �Ӷ�ȡ���ĵ����ļ������ȡָ�������ĵ�
        point_set = point_set[choice, :]

        # ŷ����
        euler_angle = point_set[:, 3:6]

        # �ڽ���
        edge_nearby = point_set[:, 6]

        # ��Ԫ����
        meta_type = point_set[:, 7]

        # ������
        point_set = point_set[:, :3]

        # �ȼ�ȥƽ���㣬�ٳ������룬������λ�úʹ�С�Ĺ�һ��
        # center # np.mean() ����һ��1*x�ľ���axis=0 ��ÿ�еľ�ֵ��axis=1����ÿ�еľ�ֵ
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # ʵ���Ƿ��expand_dimsЧ��һ��
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # �����ת��������̬�ֲ�����
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter # ���з�������̬�ֲ������

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
        ����
        [0]�㼯 (2048, 3) <class 'numpy.ndarray'>
        [1]��� (1,) <class 'numpy.ndarray'>
        [2]�ָ� (2048,) <class 'numpy.ndarray'>
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
    ��ȡ dir_path ������һ�����ļ���
    �������ļ���������������·��
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    ��ȡdir_path�µ�ȫ���ļ�·��
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
    ��һ�� predict���ڶ��� target
    :param file_name:
    :return:
    '''
    array_from_file = np.loadtxt(file_name, dtype=int)

    # ȷ������Ĵ�С���������ֵΪ5����˾����СΪ6x6��
    matrix_size = array_from_file.max() + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # ���� list1 �� list2 �����¾���
    for i in range(array_from_file.shape[1]):
        x = array_from_file[0, i]
        y = array_from_file[1, i]
        matrix[x, y] += 1

    # ��ӡ�����Բ鿴���
    print("����")
    print(matrix)

    # ʹ�� Matplotlib ���ӻ�����
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.show()


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    # ȷ������Ĵ�С���������ֵΪ5����˾����СΪ6x6��
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    # ���� list1 �� list2 �����¾���
    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    # ʹ�� Matplotlib ���ӻ�����
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
    # ��ȡ����ͼƬ�ļ�·��
    images = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(('png', 'jpg', 'jpeg'))]

    # ��ͼƬ���洢��һ���б���
    frames = [Image.open(image) for image in images]

    # ��ͼƬ����ΪGIF
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)


def search_fileall(dir_path):
    '''
    ��ȡĳ���ļ�����ȫ���ļ�·��
    :param dir_path: Ŀ���ļ���
    :return: ȫ��·������
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            filepath_all.append(os.path.join(root, file))

    return filepath_all


def metatype_statistic():
    # ���ҵ������ļ�
    dirpath = r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud'
    all_files = search_fileall(dirpath)[2:]

    all_metatype = []
    for filepath in all_files:
        point_set = np.loadtxt(filepath)
        all_metatype.append(point_set[:, 7])

    all_metatype = np.concatenate(all_metatype)
    # ʹ��unique������ȡΨһ��Ԫ�ؼ������
    unique_elements, counts = np.unique(all_metatype, return_counts=True)

    # ���ͳ�ƽ��
    for element, count in zip(unique_elements, counts):
        print(f"���� {element} ������ {count} ��")


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
    # �ҵ�ȫ���ָ��ļ�
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
            # ��ȡ�ļ��е�ÿһ�У�?��ȥ����β�Ļ��з�
            numbers = [int(line.strip()) for line in file]

            if 7 in numbers:
                print('����7���ļ���', c_file)

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
            # ��ȡ�ļ��е�ÿһ�У�?��ȥ����β�Ļ��з�
            numbers = [int(line.strip()) for line in file]

            for number in numbers:
                count_dict[number] += 1
                # if number == 7:
                #     print('����7', c_file_path)
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
