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
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode') # 是否使用CPU
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # 指定的GPU设备
    parser.add_argument('--bs', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--epoch', default=30, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training') # 优化器

    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数, [0-13)共13种

    parser.add_argument('--save_str', type=str, default='cst_pcd_abc25t', help='dataloader workers')

    parser.add_argument('--abc_pack', type=int, default=-1, help='dataloader workers')

    parser.add_argument('--rotate', default=0, type=float, help='---')

    parser.add_argument('--is_train', default='True', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
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
    # os.environ[‘CUDA_VISIBLE_DEVICES‘] 使用指定的GPU及GPU显存
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 定义数据集，训练集及对应加载器
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    abc_pack = args.abc_pack
    if not abc_pack == -1:
        print(Fore.BLACK + Back.GREEN + f'execute ABC pack trans to pack {abc_pack}')
        data_root = str(data_root).replace('{pack_idx}', str(abc_pack))

    # train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=True, is_train=True, is_back_addattr=True)
    # train_dataset = STEPMillionDataLoader(root=data_root, npoints=args.num_point, data_augmentation=True)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=5)  # , drop_last=True

    train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=False, is_back_addattr=True, is_load_all=True)
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
    if not args.use_cpu:
        predictor = predictor.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate, # 0.001
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    num_batch = len(train_dataloader)

    '''TRANING'''
    if args.is_train == 'True':
        is_train = True
    else:
        is_train = False

    if is_train:
        print(Fore.BLACK + Back.GREEN + 'training mode')

        for epoch in range(args.epoch):
            print(f'current epoch: {epoch}/{args.epoch}')
            predictor = predictor.train()

            for batch_id, data in enumerate(train_dataloader, 0):
                xyz, eula_angle_label, nearby_label, meta_type_label = data[0], data[-3], data[-2], data[-1]
                bs, n_points, _ = xyz.size()
                n_items_batch = bs * n_points

                if not args.use_cpu:
                    xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()
                else:
                    xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float(), eula_angle_label.float(), nearby_label.long(), meta_type_label.long()

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

                log_str = f'train_loss\t{loss_all.item()}\teula_loss\t{loss_eula.item()}\tnearby_loss\t{loss_nearby.item()}\tmetatype_loss\t{loss_metatype.item()}\tnearby_accu\t{correct_nearby.item() / float(n_items_batch)}\tmeta_type_accu\t{correct_meta_type.item() / float(n_items_batch)}'
                logger.info(log_str)

                prit_loss_MAD = loss_all.item()
                prit_acc_ADJ = correct_nearby.item() / float(n_items_batch)
                prit_acc_PMT = correct_meta_type.item() / float(n_items_batch)

                print_str = f'[{epoch}: {batch_id}/{num_batch}] MAD MSE loss: {prit_loss_MAD}, Acc.ADJ: {prit_acc_ADJ}, Acc.PMT: {prit_acc_PMT}'
                print(print_str)

            scheduler.step()
            torch.save(predictor.state_dict(), model_savepth)

    else:
        print(Fore.BLACK + Back.GREEN + 'eval mode')

        with torch.no_grad():
            predictor = predictor.eval()
            euler_loss_set = []
            nearby_acc_set = []
            meta_acc_set = []

            for batch_id, data in enumerate(train_dataloader, 0):
                xyz, eula_angle_label, nearby_label, meta_type_label = data[0], data[-3], data[-2], data[-1]
                bs, n_points, _ = xyz.size()
                n_items_batch = bs * n_points

                if not args.use_cpu:
                    xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float().cuda(), eula_angle_label.float().cuda(), nearby_label.long().cuda(), meta_type_label.long().cuda()
                else:
                    xyz, eula_angle_label, nearby_label, meta_type_label = xyz.float(), eula_angle_label.float(), nearby_label.long(), meta_type_label.long()

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

                euler_loss = loss_eula.item()
                nearby_acc = correct_nearby.item() / float(n_items_batch)
                meta_acc = correct_meta_type.item() / float(n_items_batch)

                euler_loss_set.append(euler_loss)
                nearby_acc_set.append(nearby_acc)
                meta_acc_set.append(meta_acc)

                log_str = f'eula_loss\t{euler_loss}\tnearby_accu\t{nearby_acc}\tmeta_type_accu\t{meta_acc}'
                logger.info(log_str)

                print_str = f'[eula loss: {euler_loss}, nearby accu: {nearby_acc}, meta type accu: {meta_acc}'
                print(print_str)

            euler_loss_all = np.mean(euler_loss_set)
            nearby_acc_all = np.mean(nearby_acc_set)
            meta_acc_all = np.mean(meta_acc_set)

            log_str = f'eula_loss_all\t{euler_loss_all}\tnearby_accu_all\t{nearby_acc_all}\tmeta_type_accu_all\t{meta_acc_all}'
            logger.info(log_str)

            print_str = f'[eula loss all: {euler_loss_all}, nearby accu all: {nearby_acc_all}, meta type accu all: {meta_acc_all}'
            print(print_str)

            tres = 0.02
            if abs(euler_loss_all - 0.0717) < tres and abs(nearby_acc_all - 0.8555) < tres and abs(meta_acc_all - 0.8646) < tres:
                print(Fore.BLACK + Back.GREEN + f'end training res: MAD-{euler_loss_all}, ADJ-{nearby_acc_all}, PMT-{meta_acc_all}')
                exit(0)


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
