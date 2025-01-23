
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
from models.CrossAttention_Cls import CrossAttention_Cls
from models.TriFeaPred_OrigValid import TriFeaPred_OrigValid


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # 指定的GPU设备
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training') # batch_size
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training') # 训练的epoch数
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training') # 学习率
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--n_metatype', type=int, default=4, help='number of considered meta type')  # 计算约束时考虑的基元数
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量

    parser.add_argument('--is_use_pred_addattr', type=str, default='False', choices=['True', 'False'], help='---') # 点数量
    parser.add_argument('--save_str', type=str, default='ca_final_e20000_labelattr', help='---') # 点数量

    parser.add_argument('--root_dataset', type=str, default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000', help='root of dataset')

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


def accuracy_over_class(all_labels: list, all_preds: list, n_classes: int):
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    accuracies = []

    for class_idx in range(n_classes):
        # 找到当前类别的所有样本
        class_mask = (all_labels == class_idx)
        # 如果当前类别的样本数为0，则跳过
        if class_mask.sum().item() == 0:
            continue
        # 计算当前类别的准确率
        class_accuracy = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
        accuracies.append(class_accuracy)

    # 返回所有类别准确率的平均值
    return np.mean(accuracies)


def mean_average_precision(all_labels: list, all_preds: list, num_class: int):
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [total_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [total_samples]

    # 将真实标签转化为one-hot编码 (one-vs-rest)
    all_labels_bin = label_binarize(all_labels, classes=np.arange(num_class))

    # 计算每个类别的AP
    ap_scores = []
    for i in range(num_class):
        ap = average_precision_score(all_labels_bin[:, i], all_preds[:, i])
        ap_scores.append(ap)

    # 计算mAP (所有类别的AP的平均值)
    mAP = np.mean(ap_scores)
    return mAP


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
    train_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=True, data_augmentation=False, is_back_addattr=True)
    test_dataset = MCBDataLoader(root=args.root_dataset, npoints=args.num_point, is_train=False, data_augmentation=False, is_back_addattr=True)
    num_class = len(train_dataset.classes)

    # sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=32, replacement=False)  # 随机选取 100 个样本
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    # sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=32, replacement=False)  # 随机选取 100 个样本
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 获取分类模型
    classifier = CrossAttention_Cls(num_class, args.n_metatype)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        classifier.load_state_dict(torch.load(model_savepth))
        print(Fore.GREEN + 'training from exist model: ' + model_savepth)
    except:
        print(Fore.GREEN + 'no existing model, training from scratch')

    if is_use_pred_addattr:
        try:
            predictor = TriFeaPred_OrigValid(n_points_all=args.num_point, n_metatype=args.n_metatype).cuda()
            predictor.load_state_dict(torch.load('model_trained/TriFeaPred_ValidOrig_fuse.pth'))
            predictor = predictor.eval()
            print(Fore.GREEN + 'load param attr predictor from', 'model_trained/TriFeaPred_ValidOrig_fuse.pth')
        except:
            print(Fore.GREEN + 'load param attr predictor failed')
            exit(1)

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

    best_instance_accu = -1.0

    '''TRANING'''
    for epoch in range(args.epoch):
        classifier = classifier.train()

        logstr_epoch = f'Epoch ({epoch + 1}/{args.epoch}):'

        mean_correct = []
        pred_cls = []
        target_cls = []

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):
            points = data[0].float().cuda()
            target = data[1].long().cuda()

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

            # 梯度置为零，否则梯度会累加
            optimizer.zero_grad()

            pred = classifier(points, eula_angle_label, nearby_label, meta_type_label)
            loss = F.nll_loss(pred, target)

            # 利用loss更新参数
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            pred_cls += pred_choice.tolist()
            target_cls += target.tolist()

        save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'train-{epoch}.png'))

        acc_over_class = accuracy_over_class(target_cls, pred_cls, num_class)
        logstr_trainaccu = f'\ttrain_instance_accu:\t{np.mean(mean_correct)}\ttrain_class_accu:\t{acc_over_class}'

        scheduler.step()
        torch.save(classifier.state_dict(), 'model_trained/' + save_str + '.pth')

        with torch.no_grad():
            classifier = classifier.eval()

            # 计算Acc.OverInstance
            total_correct = 0
            total_testset = 0

            # 保存ConfusionMatrix和计算Acc.OverClass
            pred_cls = []
            target_cls = []

            # 计算mAP
            all_preds = []
            all_labels = []

            for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points = data[0].float().cuda()
                target = data[1].long().cuda()

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

                pred = classifier(points, eula_angle_label, nearby_label, meta_type_label)

                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

                total_correct += correct.item()
                total_testset += points.size()[0]

                pred_cls += pred_choice.tolist()
                target_cls += target.tolist()

            save_confusion_mat(pred_cls, target_cls, os.path.join(confusion_dir, f'eval-{epoch}.png'))

            # 计算 mean average precision
            mAP = mean_average_precision(all_labels, all_preds, num_class)

            # 计算 Acc. over Instance
            acc_over_instance = total_correct / float(total_testset)

            # 计算 Acc. over Class
            acc_over_class = accuracy_over_class(target_cls, pred_cls, num_class)

            # 计算 F1-Score
            macro_f1_score = f1_score(target_cls, pred_cls, average='macro')
            weighted_f1_score = f1_score(target_cls, pred_cls, average='weighted')

            accustr = f'\ttest_instance_accuracy\t{acc_over_instance}\ttest_class_accuracy\t{acc_over_class}\ttest_F1_Score\t{macro_f1_score}\tmAP\t{mAP}\twmAP\t{weighted_f1_score}'
            logger.info(logstr_epoch + logstr_trainaccu + accustr)
            print(accustr.replace('\t', ' '))

            # 额外保存最好的模型
            if best_instance_accu < acc_over_class:
                best_instance_accu = acc_over_class
                torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    clear_log('./log')
    init(autoreset=True)
    main(parse_args())


