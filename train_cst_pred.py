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
import numpy as np

# 自定义模块
# from models.TriFeaPred import TriFeaPred
from data_utils.ParamDataLoader import ParamDataLoader
from data_utils.ParamDataLoader import MCBDataLoader

from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode') # 是否使用CPU
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # 指定的GPU设备
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]') # 已训练好的分类模型
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40') # 指定训练集 ModelNet10/40
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training') # 优化器
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数, [0-13)共13种
    parser.add_argument('--workers', type=int, default=10, help='dataloader workers')
    parser.add_argument('--save_str', type=str, default='TriFeaPred_ValidOrig', help='dataloader workers')

    parser.add_argument('--local', default='True', choices=['True', 'False'], type=str, help='---')
    parser.add_argument('--root_sever', type=str, default=r'/root/my_data/data_set/STEP20000_Hammersley_2000', help='root of dataset')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000', help='root of dataset')
    # 点云数据集根目录
    # 服务器：r'/opt/data/private/data_set/PointCloud_Xindi_V2/'

    args = parser.parse_args()
    print(args)
    return args


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main(args):
    save_str = args.save_str

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

    train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, data_augmentation=True, is_train=True, is_back_addattr=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))  # , drop_last=True

    '''MODEL LOADING'''
    predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()

    model_savepth = 'model_trained/' + save_str + '_fuse.pth'
    try:
        predictor.load_state_dict(torch.load(model_savepth))
        print('training from exist model: ' + model_savepth)
    except:
        print('no existing model, training from scratch')

    predictor.apply(inplace_relu)
    if not args.use_cpu:
        predictor = predictor.cuda()

    # args.optimizer='Adam'
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate, # 0.001
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate # 1e-4
        )
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    num_batch = len(train_dataloader)

    '''TRANING'''
    for epoch in range(args.epoch):
        print(f'current epoch: {epoch}/{args.epoch}')
        predictor = predictor.train()

        for batch_id, data in enumerate(train_dataloader, 0):
            xyz, cls, eula_angle_label, nearby_label, meta_type_label = data
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

            print_str = f'[{epoch}: {batch_id}/{num_batch}] train loss: {loss_all.item()}, eula loss: {loss_eula.item()}, nearby loss: {loss_nearby.item()},metatype loss: {loss_metatype.item()}, nearby accu: {correct_nearby.item() / float(n_items_batch)}, meta type accu: {correct_meta_type.item() / float(n_items_batch)}'
            print(print_str)

        scheduler.step()
        torch.save(predictor.state_dict(), model_savepth)


if __name__ == '__main__':
    # asas = np.loadtxt(r'D:\document\DeepLearning\DataSet\STEPMillion\STEPMillion_pack1\overall\2629.txt')
    # print(asas)

    parsed_args = parse_args()
    main(parsed_args)
