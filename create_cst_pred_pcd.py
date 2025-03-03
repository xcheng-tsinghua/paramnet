'''
训练三属性预测分支
'''
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
import torch.nn.functional as F
from datetime import datetime
import logging # 记录日志信息
import argparse
from colorama import Fore, Back, Style, init
import numpy as np

# 自定义模块
# from models.TriFeaPred import TriFeaPred
from data_utils.ParamDataLoader import ParamDataLoader
from data_utils.ParamDataLoader import MCBDataLoader, STEPMillionDataLoader

from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid as cst_pcd
# from models.hpnet import PrimitiveNet as cst_pcd
# from models.parsenet import PrimitivesEmbeddingDGCNGn as cst_pcd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=10, help='batch size in training') # batch_size
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数, [0-13)共13种

    parser.add_argument('--save_str', type=str, default='cst_pcd_abc25t', help='dataloader workers')
    parser.add_argument('--save_root', type=str, default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000_pl', help='data save root')

    parser.add_argument('--local', default='True', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str,
                        default=r'/root/my_data/data_set/STEP20000_Hammersley_2000',
                        help='root of dataset')
    parser.add_argument('--root_local', type=str,
                        default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000',
                        help='root of dataset')

    # Parametric20000
    # sever: r'/root/my_data/data_set/STEP20000_Hammersley_2000'
    # local: r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000'
    # ABC
    # sever: r'/root/my_data/data_set/STEPMillion/STEPMillion_pack{pack_idx}/overall'
    # local: r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_{pack_idx}\STEPMillion_pack{pack_idx}\overall'

    args = parser.parse_args()
    print(args)
    return args


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main(args):
    save_str = args.save_str

    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''HYPER PARAMETER'''
    # 定义数据集，训练集及对应加载器
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False, is_back_addattr=True, is_load_all=True, is_back_froot=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

    '''MODEL LOADING'''
    predictor = cst_pcd(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        predictor.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    predictor.apply(inplace_relu)
    predictor = predictor.cuda()

    '''TRANING'''

    print(Fore.BLACK + Back.GREEN + 'eval mode')

    with torch.no_grad():
        predictor = predictor.eval()

        for batch_id, data in enumerate(train_dataloader, 0):
            xyz, file_path = data[0], data[-1]
            xyz = xyz.float().cuda()

            pred_eula_angle, pred_edge_nearby, pred_meta_type = predictor(xyz)

            pred_edge_nearby = pred_edge_nearby.contiguous().view(-1, 2)
            choice_nearby = pred_edge_nearby.data.max(1)[1]

            pred_meta_type = pred_meta_type.contiguous().view(-1, args.n_metatype)
            choice_meta_type = pred_meta_type.data.max(1)[1]



def clear_log(log_dir):
    """
    清空空白的log文件
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(log_dir):
        # 获取文件的完整路径
        file_path = os.path.join(log_dir, filename)
        # 检查是否为txt文件且为空
        if filename.endswith('.txt') and os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")


if __name__ == '__main__':
    # asas = np.loadtxt(r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_pack1\overall\2629.txt')
    # print(asas)

    parsed_args = parse_args()
    # clear_log('./log')
    init(autoreset=True)
    main(parsed_args)
