"""
显示360Gallery的分割效果图
"""

import os
import sys
# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# 将models文件夹的路径添加到sys.path中，使得models文件夹中的py文件能被本文件import
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

# 工具包
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import logging # 记录日志信息
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 自建模块
from data_utils.ParamDataLoader import Seg360GalleryDataLoader
from data_utils.ParamDataLoader import segfig_save
from models.CrossAttention_Seg import CrossAttention_Seg
from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--save_str', type=str, default='best_ca_final_predattr_part_seg', help='save string')
    parser.add_argument('--is_use_pred_addattr', type=str, default='True', choices=['True', 'False'], help='--')

    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量
    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\360Gallery_Seg', help='root of dataset')

    # 参数化数据集：D:/document/DeepLearning/DataSet/data_set_p2500_n10000
    # 参数化数据集(新迪)：r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud'
    # 参数化数据集(新迪，服务器)：r'/opt/data/private/data_set/PointCloud_Xindi_V2/'
    # modelnet40数据集：r'D:\document\DeepLearning\DataSet\modelnet40_normal_resampled'
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_B
    # 360 Gallery: D:\document\DeepLearning\DataSet\360Gallery_Seg

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def batch_segfig_save(points, seg_pred, save_path, is_gt, save_step):
    bs = points.size()[0]

    for idx, i in enumerate(range(bs)):

        if idx % save_step == 0:
            plt.axis('off')
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            points_show = points[i].cpu().numpy().copy()
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
            c_data = seg_pred[i].max(dim=1)[1].cpu().numpy()

            c_data_trans = []

            for cc_data in c_data:
                c_val = cc_data.item()

                if c_val == 0:
                    c_data_trans.append((156 / 255, 64 / 255, 132 / 255))
                elif c_val == 1:
                    c_data_trans.append((110 / 255, 189 / 255, 183 / 255))
                else:
                    c_data_trans.append((63 / 255, 129 / 255, 180 / 255))

            ax.scatter(x_data, y_data, z_data, c=c_data_trans, s=100, edgecolors='none')

            save_name, save_ext = os.path.splitext(save_path)

            if is_gt:
                final_save = save_name + str(i) + 'GT' + save_ext
            else:
                final_save = save_name + str(i) + save_ext

            plt.savefig(final_save)
            plt.close()


def main(args):
    save_str = args.save_str
    if args.is_use_pred_addattr == 'True':
        is_use_pred_addattr = True
    else:
        is_use_pred_addattr = False

    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)

    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 定义数据集，训练集及对应加载器
    train_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=True)
    test_dataset = Seg360GalleryDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=False)
    n_segpart = len(test_dataset.seg_names) # num_classes 数据集中的模型类别数
    print('num of segment part: ', n_segpart)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 获取分类模型
    classifier = CrossAttention_Seg(n_segpart, args.n_metatype)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, can not predict')
        exit(1)

    if is_use_pred_addattr:
        try:
            predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
            predictor.load_state_dict(torch.load('model_trained/TriFeaPred_ValidOrig_fuse.pth'))
            predictor = predictor.eval()
            print('load param attr predictor from', 'model_trained/TriFeaPred_ValidOrig_fuse.pth')
        except:
            print('load param attr predictor failed')
            exit(1)

    classifier.apply(inplace_relu)
    classifier = classifier.cuda()
    classifier = classifier.eval()

    with torch.no_grad():

        save_fig_count = 0

        # 保存间隔
        save_step = 20

        for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

            if batch_id % save_step == 0:

                points = data[0].float().cuda()
                target = data[1].long().cuda()

                # 使用预测属性
                if is_use_pred_addattr:
                    euler_angle, nearby, meta_type = predictor(points)
                    nearby, meta_type = torch.exp(nearby), torch.exp(meta_type)
                    euler_angle, nearby, meta_type = euler_angle.detach(), nearby.detach(), meta_type.detach()

                else:
                    euler_angle = data[2].float().cuda()
                    nearby = data[3].long().cuda()
                    meta_type = data[4].long().cuda()

                    # 将标签转化为 one-hot
                    nearby = F.one_hot(nearby, 2)
                    meta_type = F.one_hot(meta_type, args.n_metatype)

                seg_pred = classifier(points, euler_angle, nearby, meta_type)

                save_path = os.path.join(confusion_dir, f'test-{save_fig_count}.png')

                batch_save_step = 3
                batch_segfig_save(points, seg_pred, save_path, False, batch_save_step)
                batch_segfig_save(points, F.one_hot(target, n_segpart), save_path, True, batch_save_step)

                save_fig_count += 1


if __name__ == '__main__':
    main(parse_args())





