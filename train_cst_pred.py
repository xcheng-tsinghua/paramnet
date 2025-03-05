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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bs', type=int, default=25, help='batch size in training') # batch_size
    parser.add_argument('--epoch', default=25, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数, [0-13)共13种

    parser.add_argument('--is_load_weight', type=str, default='True', choices=['True', 'False'], help='---')
    parser.add_argument('--model', type=str, default='hpnet', choices=['hpnet', 'parsenet', 'cstpcd'], help='model used for pred')
    parser.add_argument('--save_str', type=str, default='parsenet', help='dataloader workers')  # cst_pcd_abc25t

    # parser.add_argument('--is_train', default='True', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--abc_pack', type=int, default=-1, help='pack of abc')  # 点数量
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

    if args.abc_pack != -1:
        print('use abc dataset')
        train_root = data_root.replace('{pack_idx}', str(args.abc_pack))

        train_dataset = STEPMillionDataLoader(root=train_root, npoints=args.num_point, data_augmentation=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

        test_root = data_root.replace('{pack_idx}', str(21))
        test_dataset = STEPMillionDataLoader(root=test_root, npoints=args.num_point, data_augmentation=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

    else:
        train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False, is_back_addattr=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True
        test_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False, is_back_addattr=True, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

    '''MODEL LOADING'''
    print(Fore.BLACK + Back.BLUE + f'pred model: {args.model}')
    if args.model == 'cstpcd':
        predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
    elif args.model == 'hpnet':
        predictor = PrimitiveNet(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
    elif args.model == 'parsenet':
        predictor = PrimitivesEmbeddingDGCNGn(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
    else:
        raise TypeError('error model name!')

    model_savepth = 'model_trained/' + save_str + '.pth'
    if args.is_load_weight == 'True':
        try:
            predictor.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    predictor.apply(inplace_relu)

    predictor = predictor.cuda()


    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=args.learning_rate, # 0.001
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''TRANING'''
    for epoch in range(args.epoch):
        print(f'current epoch: {epoch}/{args.epoch}')
        predictor = predictor.train()

        acc_mad = []
        acc_adj = []
        acc_pmt = []

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            xyz, eula_angle_label, nearby_label, meta_type_label = data[0], data[-3], data[-2], data[-1]
            bs, n_points, _ = xyz.size()
            n_items_batch = bs * n_points

            xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()

            optimizer.zero_grad()
            pred_eula_angle, pred_edge_nearby, pred_meta_type = predictor(xyz)

            # vis_pointcloudattr(point_set[0, :, :].detach().cpu().numpy(), np.argmax(pred_meta_type[0, :, :].detach().cpu().numpy(), axis=1))

            loss_eula = F.mse_loss(eula_angle_label, pred_eula_angle)

            pred_edge_nearby = pred_edge_nearby.contiguous().view(-1, 2)
            nearby_label = nearby_label.view(-1)
            loss_nearby = F.nll_loss(pred_edge_nearby, nearby_label)

            pred_meta_type = pred_meta_type.contiguous().view(-1, args.n_metatype)
            meta_type_label = meta_type_label.view(-1)
            loss_metatype = F.nll_loss(pred_meta_type, meta_type_label)

            loss_all = loss_eula + loss_nearby + loss_metatype

            loss_all.backward()
            optimizer.step()

            # accu
            choice_nearby = pred_edge_nearby.data.max(1)[1]
            correct_nearby = choice_nearby.eq(nearby_label.data).cpu().sum()
            choice_meta_type = pred_meta_type.data.max(1)[1]
            correct_meta_type = choice_meta_type.eq(meta_type_label.data).cpu().sum()

            # log_str = f'train_loss\t{loss_all.item()}\teula_loss\t{loss_eula.item()}\tnearby_loss\t{loss_nearby.item()}\tmetatype_loss\t{loss_metatype.item()}\tnearby_accu\t{correct_nearby.item() / float(n_items_batch)}\tmeta_type_accu\t{correct_meta_type.item() / float(n_items_batch)}'
            # logger.info(log_str)

            prit_loss_MAD = loss_all.detach().item()
            prit_acc_ADJ = correct_nearby.item() / float(n_items_batch)
            prit_acc_PMT = correct_meta_type.item() / float(n_items_batch)

            acc_mad.append(prit_loss_MAD)
            acc_adj.append(prit_acc_ADJ)
            acc_pmt.append(prit_acc_PMT)

            # print_str = f'[{epoch}: {batch_id}/{num_batch}] MAD MSE loss: {prit_loss_MAD}, Acc.ADJ: {prit_acc_ADJ}, Acc.PMT: {prit_acc_PMT}'
            # print(print_str)

        train_acc_mad = np.mean(acc_mad)
        train_acc_adj = np.mean(acc_adj)
        train_acc_pmt = np.mean(acc_pmt)

        scheduler.step()
        print('save model to:' + model_savepth)
        torch.save(predictor.state_dict(), model_savepth)

        with torch.no_grad():
            predictor = predictor.eval()

            acc_mad = []
            acc_adj = []
            acc_pmt = []

            for batch_id, data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)):
                xyz, eula_angle_label, nearby_label, meta_type_label = data[0], data[-3], data[-2], data[-1]
                bs, n_points, _ = xyz.size()
                n_items_batch = bs * n_points

                xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()

                pred_eula_angle, pred_edge_nearby, pred_meta_type = predictor(xyz)

                loss_eula = F.mse_loss(eula_angle_label, pred_eula_angle)
                pred_edge_nearby = pred_edge_nearby.contiguous().view(-1, 2)
                nearby_label = nearby_label.view(-1)

                pred_meta_type = pred_meta_type.contiguous().view(-1, args.n_metatype)
                meta_type_label = meta_type_label.view(-1)

                # accu
                choice_nearby = pred_edge_nearby.data.max(1)[1]
                correct_nearby = choice_nearby.eq(nearby_label.data).cpu().sum()
                choice_meta_type = pred_meta_type.data.max(1)[1]
                correct_meta_type = choice_meta_type.eq(meta_type_label.data).cpu().sum()

                euler_loss = loss_eula.detach().item()
                nearby_acc = correct_nearby.item() / float(n_items_batch)
                meta_acc = correct_meta_type.item() / float(n_items_batch)

                acc_mad.append(euler_loss)
                acc_adj.append(nearby_acc)
                acc_pmt.append(meta_acc)

                # log_str = f'eula_loss\t{euler_loss}\tnearby_accu\t{nearby_acc}\tmeta_type_accu\t{meta_acc}'
                # logger.info(log_str)
                #
                # print_str = f'[eula loss: {euler_loss}, nearby accu: {nearby_acc}, meta type accu: {meta_acc}'
                # print(print_str)

            test_acc_mad = np.mean(acc_mad)
            test_acc_adj = np.mean(acc_adj)
            test_acc_pmt = np.mean(acc_pmt)

            log_str = f'{epoch}/{args.epoch}\ttrain_mad_adj_pmt\t{train_acc_mad}\t{train_acc_adj}\t{train_acc_pmt}\ttest_mad_adj_pmt\t{test_acc_mad}\t{test_acc_adj}\t{test_acc_pmt}'
            logger.info(log_str)

            print(log_str.replace('\t', ' '))


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
