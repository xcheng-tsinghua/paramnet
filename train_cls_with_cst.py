"""
带约束的训练
包含预测约束与label约束
"""


"""
用于训练目前论文中确定的模型
Cross Attention 分类训练脚本
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
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from colorama import Fore, Back, Style, init

# 自建模块
from data_utils.ParamDataLoader import MCBDataLoader
from data_utils.ParamDataLoader import save_confusion_mat
from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid
from models.GCN3D import GCN3D
from models.dgcnn import DGCNN
from models.PointNet import PointNet
from models.PointConv import PointConv
from models.PointCNN import PointCNN
from models.PointNet2 import PointNet2
from models.PPFNet import PPFNet
from models.CrossAttention_Cls import CrossAttention_Cls as model_cls
from data_utils.ParamDataLoader import all_metric_cls, PrismCuboidDataLoader


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # 指定的GPU设备
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--epoch', default=35, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量

    parser.add_argument('--is_use_pred_addattr', type=str, default='True', choices=['True', 'False'], help='---') # 点数量
    parser.add_argument('--save_str', type=str, default='ca_final', help='---')
    parser.add_argument('--rotate', default=0, type=int, help='---')
    parser.add_argument('--cst_pcd', type=str, default='cst_pcd_abc25t.pth', help='---')
    parser.add_argument('--model', type=str, default='PointNet2', choices=['GCN3D', 'DGCNN', 'PointNet', 'PointNet2'], help='model used for cls')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str,
                        default=r'/root/my_data/data_set/STEP20000_Hammersley_2000',
                        help='root of dataset')
    parser.add_argument('--root_local', type=str,
                        default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000',
                        help='root of dataset')

    # 参数化数据集：D:/document/DeepLearning/DataSet/data_set_p2500_n10000
    # 参数化数据集(新迪)：r'D:\document\DeepLearning\ParPartsNetWork\dataset_xindi\pointcloud'
    # 参数化数据集(新迪，服务器)：r'/opt/data/private/data_set/PointCloud_Xindi_V2/'
    # modelnet40数据集：r'D:\document\DeepLearning\DataSet\modelnet40_normal_resampled'
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_A
    # MCB 数据集：D:\document\DeepLearning\DataSet\MCB_PointCloud\MCBPcd_B
    # Engineering20000数据集：D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000
    # Engineering20000数据集，服务器：/root/my_data/data_set/STEP20000_Hammersley_2000

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    if args.is_use_pred_addattr == 'True':
        is_use_pred_addattr = True
        print(Fore.GREEN + 'use predict parametric attribute')
    else:
        is_use_pred_addattr = False
        print(Fore.GREEN + 'use label parametric attribute')

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 定义数据集，训练集及对应加载器
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, is_train=True, data_augmentation=False, is_back_addattr=True, rotate=args.rotate)
    test_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, is_train=False, data_augmentation=False, is_back_addattr=True, rotate=args.rotate)
    num_class = len(train_dataset.classes)

    # sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=32, replacement=False)  # 随机选取 100 个样本
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    # sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=32, replacement=False)  # 随机选取 100 个样本
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 获取分类模型
    if args.model == 'GCN3D':
        classifier = GCN3D(support_num=1, neighbor_num=20, classes_num=num_class, fea_channel=9).cuda()
    elif args.model == 'DGCNN':
        classifier = DGCNN(output_channels=num_class, fea_channel=9).cuda()
    elif args.model == 'PointNet':
        classifier = PointNet(k=num_class, fea_channel=9).cuda()
    elif args.model == 'PointNet2':
        classifier = PointNet2(num_class=num_class, fea_channel=9).cuda()
    else:
        raise TypeError('error model name!')

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print(Fore.GREEN + 'training from exist model: ' + model_savepth)
    except:
        print(Fore.GREEN + 'no existing model, training from scratch')

    if is_use_pred_addattr:
        try:
            predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
            predictor.load_state_dict(torch.load('model_trained/' + args.cst_pcd))
            predictor = predictor.eval()
            print(Fore.GREEN + 'load param attr predictor from', 'model_trained/', args.cst_pcd)
        except:
            print(Fore.GREEN + 'load param attr predictor failed')
            exit(1)
    else:
        print(Fore.GREEN + 'use label attr')

    classifier.apply(inplace_relu)
    classifier = classifier.cuda()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate, # 0.001
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate # 1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    best_instance_accu = -1.0
    for epoch in range(args.epoch):
        classifier = classifier.train()

        logstr_epoch = f'Epoch({epoch}/{args.epoch}):'
        all_preds = []
        all_labels = []

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            points, target = data[0].float().cuda(), data[1].long().cuda()

            # 使用预测属性
            if is_use_pred_addattr:
                eula_angle_label, nearby_label, meta_type_label = predictor(points)
                nearby_label, meta_type_label = torch.exp(nearby_label), torch.exp(meta_type_label)
                eula_angle_label, nearby_label, meta_type_label = eula_angle_label.detach(), nearby_label.detach(), meta_type_label.detach()

            else:
                eula_angle_label = data[2].float().cuda()
                nearby_label = data[3].long().cuda()
                meta_type_label = data[4].long().cuda()

                # 将标签转化为 one-hot
                nearby_label = F.one_hot(nearby_label, 2)
                meta_type_label = F.one_hot(meta_type_label, args.n_metatype)

            cstattr = torch.cat([eula_angle_label, nearby_label, meta_type_label], dim=-1).permute(0, 2, 1)

            # -> [bs, 3, n_points]
            points = points.permute(0, 2, 1)
            assert points.size()[1] == 3

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            pred = classifier(points, cstattr)
            loss = F.nll_loss(pred, target)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            # 保存数据用于计算指标
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

        # 计算分类指标
        all_metric_train = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'train-{epoch}.png'))
        logstr_trainaccu = f'\ttrain_instance_accu:\t{all_metric_train[0]}'

        # 调整学习率并保存权重
        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        '''测试'''
        with torch.no_grad():
            classifier = classifier.eval()

            all_preds = []
            all_labels = []

            for j, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                points, target = data[0].float().cuda(), data[1].long().cuda()

                # 使用预测属性
                if is_use_pred_addattr:
                    eula_angle_label, nearby_label, meta_type_label = predictor(points)
                    nearby_label, meta_type_label = torch.exp(nearby_label), torch.exp(meta_type_label)
                    eula_angle_label, nearby_label, meta_type_label = eula_angle_label.detach(), nearby_label.detach(), meta_type_label.detach()

                else:
                    eula_angle_label = data[2].float().cuda()
                    nearby_label = data[3].long().cuda()
                    meta_type_label = data[4].long().cuda()

                    # 将标签转化为 one-hot
                    nearby_label = F.one_hot(nearby_label, 2)
                    meta_type_label = F.one_hot(meta_type_label, args.n_metatype)

                cstattr = torch.cat([eula_angle_label, nearby_label, meta_type_label], dim=-1).permute(0, 2, 1)

                points = points.permute(0, 2, 1)
                assert points.size()[1] == 3

                pred = classifier(points, cstattr)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

            all_metric_eval = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'eval-{epoch}.png'))
            accustr = f'\teval_ins_acc\t{all_metric_eval[0]}\teval_cls_acc\t{all_metric_eval[1]}\teval_f1_m\t{all_metric_eval[2]}\teval_f1_w\t{all_metric_eval[3]}\tmAP\t{all_metric_eval[4]}'
            logger.info(logstr_epoch + logstr_trainaccu + accustr)

            print(f'{save_str}: epoch {epoch}/{args.epoch}: train_ins_acc: {all_metric_train[0]}, test_ins_acc: {all_metric_eval[0]}')

            # 额外保存最好的模型
            if best_instance_accu < all_metric_eval[0]:
                best_instance_accu = all_metric_eval[0]
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    # clear_log('./log')
    init(autoreset=True)
    main(parse_args())




