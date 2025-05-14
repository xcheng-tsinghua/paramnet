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
from tqdm import tqdm

# 自定义模块
# from models.TriFeaPred import TriFeaPred
from data_utils.ParamDataLoader import ParamDataLoader
from data_utils.ParamDataLoader import MCBDataLoader, STEPMillionDataLoader

from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid
from models.hpnet import PrimitiveNet
from models.parsenet import PrimitivesEmbeddingDGCNGn
from vis.vis import view_pcd_paper, ex_paras, vis_mesh_view


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bs', type=int, default=2, help='batch size in training') # batch_size
    parser.add_argument('--num_point', type=int, default=2800, help='Point Number') # 点数量
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数, [0-13)共13种

    parser.add_argument('--model', type=str, default='cstpcd', choices=['hpnet', 'parsenet', 'cstpcd'], help='model used for pred')
    parser.add_argument('--save_str', type=str, default='cstpcd_abc', help='dataloader workers')  # cst_pcd_abc25t

    parser.add_argument('--is_vis', default='True', choices=['True', 'False'], type=str, help='whether vis pred cst')
    parser.add_argument('--dataset', default='mcb', choices=['mcb', 'abc', 'p20'], type=str, help='---')

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
    if args.dataset == 'abc':
        data_root = r'D:\document\DeepLearning\paper_draw\AttrVis_ABC'
    elif args.dataset == 'mcb':
        data_root = r'D:\document\DeepLearning\paper_draw\AttrVis_MCB2'
    elif args.dataset == 'p20':
        data_root = r'F:\document\deeplearning\Param20000_Distributed\test'
    else:
        raise TypeError('error dataset')

    if args.dataset == 'mcb':
        test_dataset = STEPMillionDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False,
                                         is_backaddattr=False, noise_scale=0.004)
    else:
        test_dataset = STEPMillionDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False, is_backaddattr=True, noise_scale=0.004)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

    '''MODEL LOADING'''
    hpnet = PrimitiveNet(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
    parsenet = PrimitivesEmbeddingDGCNGn(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
    cstpcd = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()

    hpnet.load_state_dict(torch.load('model_trained/hpnet_abc.pth'))
    parsenet.load_state_dict(torch.load('model_trained/parsenet_abc.pth'))
    cstpcd.load_state_dict(torch.load('model_trained/cstpcd_abc.pth'))

    # predictor.apply(inplace_relu)
    # predictor = predictor.cuda()

    '''evaluating'''
    with torch.no_grad():
        hpnet = hpnet.eval()
        parsenet = parsenet.eval()
        cstpcd = cstpcd.eval()

        for batch_id, data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)):

            if args.dataset == 'mcb':
                xyz, xyz_n, data_idx = data[0], data[1], data[-1]
                xyz, xyz_n = xyz.float().cuda(), xyz_n.float().cuda()
            else:
                xyz, xyz_n, eula_angle_label, nearby_label, meta_type_label, data_idx = data[0], data[1], data[2], data[3], data[4], data[5]
                xyz, xyz_n, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), xyz_n.float().cuda(),eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()

            h_mad, h_adj, h_pmt = hpnet(xyz_n)
            p_mad, p_adj, p_pmt = parsenet(xyz_n)
            c_mad, c_adj, c_pmt = cstpcd(xyz)

            bs_vis = 0
            mesh_path = test_dataset.get_data_path(data_idx[bs_vis].item())

            if args.dataset == 'abc':
                mesh_path = mesh_path.replace('.txt', '.stl')
            elif args.dataset == 'mcb':
                mesh_path = mesh_path.replace('.txt', '.obj')
            elif args.dataset == 'p20':
                mesh_path = mesh_path.replace('.txt', '.stl')
            else:
                raise TypeError('error dataset')

            if args.dataset != 'mcb':
                _, l_mad, l_adj, l_pmt = ex_paras(xyz, eula_angle_label, nearby_label, meta_type_label, bs_vis, is_label=True)

            _, h_mad, h_adj, h_pmt = ex_paras(xyz, h_mad, h_adj, h_pmt, bs_vis)
            _, p_mad, p_adj, p_pmt = ex_paras(xyz, p_mad, p_adj, p_pmt, bs_vis)
            coor, c_mad, c_adj, c_pmt = ex_paras(xyz, c_mad, c_adj, c_pmt, bs_vis)

            # 显示mesh
            vis_mesh_view(mesh_path, is_save_view=True)

            # 显示label
            if args.dataset != 'mcb':
                view_pcd_paper(coor, l_mad, is_save_view=True)
                view_pcd_paper(coor, l_adj, is_save_view=True)
                view_pcd_paper(coor, l_pmt, is_save_view=True)

            # 显示cstnet
            view_pcd_paper(coor, c_mad, is_save_view=True)
            view_pcd_paper(coor, c_adj, is_save_view=True)
            view_pcd_paper(coor, c_pmt, is_save_view=True)

            # 显示parsenet
            view_pcd_paper(coor, p_mad, is_save_view=True)
            view_pcd_paper(coor, p_adj, is_save_view=True)
            view_pcd_paper(coor, p_pmt, is_save_view=True)

            # 显示hpnet
            view_pcd_paper(coor, h_mad, is_save_view=True)
            view_pcd_paper(coor, h_adj, is_save_view=True)
            view_pcd_paper(coor, h_pmt, is_save_view=True)


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
