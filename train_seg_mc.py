
"""
Author: Benny
Date: Nov 2019
"""
import os
import sys
# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# 将models文件夹的路径添加到sys.path中，使得models文件夹中的py文件能被本文件import
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

import argparse
import torch
from datetime import datetime
import logging
import sys
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from data_utils.ParamDataLoader import ShapeNet50SegDataLoader
from models.CrossAttention_Seg_MClass import CrossAttention_Seg_MClass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
               # ,'add': [50, 51, 52, 53, 54, 55, 56, 57, 58]
                # 'add': [0, 1, 2, 3, 4]
               }
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


import matplotlib.pyplot as plt


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """
        num_classes: vector length
        y: one-hot location
        e.g. y=3, num_class=5 return (0, 0, 1, 0, 0)
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=50, type=int, help='epoch to run') # 5
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path') # pointnet2_part_seg_msg
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals') # action: 仅输入变量名为输入取值时的行为
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--root', type=str, default=r'D:\document\DeepLearning\DataSet\shapenetcore_partanno_segmentation_benchmark_v0_normal', help='root of dataset')

    return parser.parse_args()


def compute_instance_miou(seg_pred, target, n_seg_part):
    """
    计算单个 batch 的 Instance MIOU
    Args:
        seg_pred: torch.Tensor, 大小为 [batch_size, n_points, n_seg_part]
        target: torch.Tensor, 大小为 [batch_size, n_points]
        n_seg_part: int, 分割的部分数量
    Returns:
        Instance MIOU: float
    """
    batch_size = seg_pred.shape[0]

    # 1. 获取每个点的预测分割部分
    seg_pred_label = torch.argmax(seg_pred, dim=-1)  # [batch_size, n_points]

    iou_list = []

    # 2. 对每个 batch 中的实例分别计算 IOU
    for b in range(batch_size):
        iou_per_part = []
        for part in range(n_seg_part):
            # 计算预测和标签中属于当前 part 的点
            pred_mask = (seg_pred_label[b] == part)  # 预测属于该 part 的点
            target_mask = (target[b] == part)  # 真实属于该 part 的点

            # 计算交集（TP）和并集（TP + FP + FN）
            intersection = torch.sum(pred_mask & target_mask).item()
            union = torch.sum(pred_mask | target_mask).item()

            if union == 0:
                # 如果并集为 0，说明该 part 在预测和标签中都不存在，跳过
                continue
            else:
                iou = intersection / union  # 计算 IOU
                iou_per_part.append(iou)

        if iou_per_part:
            # 计算该实例的平均 IOU
            instance_miou = sum(iou_per_part) / len(iou_per_part)
            iou_list.append(instance_miou)

    if iou_list:
        # 计算整个 batch 的平均 IOU
        return sum(iou_list) / len(iou_list)
    else:
        return 0.0


def main(args):
    save_str = 'ca_final_predattr_shapenet'

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    TRAIN_DATASET = ShapeNet50SegDataLoader(root=args.root, npoints=args.npoint, split='trainval', normal_channel=args.normal) # args.normal = Flase
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    TEST_DATASET = ShapeNet50SegDataLoader(root=args.root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''

    classifier = CrossAttention_Seg_MClass(num_part, 4).cuda()  # args.normal = False
    criterion = torch.nn.CrossEntropyLoss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    model_savepth = 'model_trained/' + save_str + '.pth'
    try:
        checkpoint = torch.load(model_savepth)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Load pretrain model from: {model_savepth}')
    except:
        log_string('No existing model, starting training from scratch...')
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        # 计算评价指标用变量
        total_correct = 0
        total_points = 0
        iou_per_part_sum = torch.zeros(num_part)
        iou_per_part_count = torch.zeros(num_part)

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            class_onehot = to_categorical(label, num_classes)
            seg_pred, trans_feat = classifier(points, class_onehot)
            pred_vis = seg_pred

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]

            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

            if is_show_img:
                if i % 10 == 0:
                    points_show = points.transpose(2, 1)[0].cpu().numpy().copy()
                    ax.clear()
                    ax.scatter(points_show[:, 0], points_show[:, 1], points_show[:, 2], c=pred_vis[0].max(dim=1)[1].cpu().numpy())
                    plt.pause(100)

            # instance mean Intersection over Union (instance mIOU)
            # 对每个分割部件计算 IoU
            pred_classes = pred_vis.argmax(dim=2).view(-1)  # Size: [batch_size, n_points]
            total_correct += (pred_classes == target).sum().item()
            total_points += target.numel()

            for part in range(num_part):
                intersection = ((pred_classes == part) & (target == part)).sum().item()
                union = ((pred_classes == part) | (target == part)).sum().item()
                if union > 0:  # 只在存在此部件时才计算 IoU
                    iou_per_part_sum[part] += float(intersection) / float(union)
                    iou_per_part_count[part] += 1

        oa = total_correct / total_points
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        for c_part in range(num_part):
            if abs(iou_per_part_count[c_part].item()) < 1e-6:
                # iou_per_part_count[c_part] = 1
                # iou_per_part_sum[c_part] = 0
                print('发现某个分割类别为零:', c_part)

        iou_per_part_avg = iou_per_part_sum / iou_per_part_count
        iou_per_part_avg = torch.nan_to_num(iou_per_part_avg, nan=0.0)
        miou = iou_per_part_avg.mean().item()

        accustr = f'test_oa\t{oa}\ttest_miou\t{miou}'
        for c_part in range(num_part):
            accustr += f'\t{c_part}_miou\t{iou_per_part_avg[c_part]}'
        print(accustr)

        with torch.no_grad():
            total_miou = 0.0  # 累加每个 batch 的 MIOU
            total_batches = 0  # 记录 batch 的数量

            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            # 计算评价指标用变量
            total_correct = 0
            total_points = 0
            iou_per_part_sum = torch.zeros(num_part)
            iou_per_part_count = torch.zeros(num_part)

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                target2 = target
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))

                batch_miou = compute_instance_miou(seg_pred, target, num_part)
                total_miou += batch_miou
                total_batches += 1

                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val  # 预测结果, -> [bs, n_point, n_seg_class]
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # -> [bs, n_point]
                target = target.cpu().data.numpy()  # -> [bs, n_point]

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]  # -> 类别，将分割类别转换为对应的物体类型名
                    logits = cur_pred_val_logits[i, :, :]  # -> 当前batch所有点的预测结果，[n_point, n_seg_class]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

                # 评价指标变量
                # Overall Accuracy (OA)
                target = target2.view(-1)
                pred_classes = seg_pred.argmax(dim=2).view(-1)  # Size: [batch_size, n_points]
                total_correct += (pred_classes == target).sum().item()
                total_points += target.numel()

                # instance mean Intersection over Union (instance mIOU)
                # 对每个分割部件计算 IoU
                for part in range(num_part):
                    intersection = ((pred_classes == part) & (target == part)).sum().item()
                    union = ((pred_classes == part) | (target == part)).sum().item()
                    if union > 0:  # 只在存在此部件时才计算 IoU
                        iou_per_part_sum[part] += float(intersection) / float(union)
                        iou_per_part_count[part] += 1

            # 计算该epoch的评价指标
            oa = total_correct / total_points

            iou_per_part_avg = iou_per_part_sum / iou_per_part_count
            iou_per_part_avg = torch.nan_to_num(iou_per_part_avg, nan=0.0)
            miou = iou_per_part_avg.mean().item()

            # 计算所有 batch 的平均 MIOU
            epoch_miou = total_miou / total_batches if total_batches > 0 else 0.0

            accustr = f'test_oa\t{oa}\ttest_miou\t{miou}\tinstance miou\t{epoch_miou}'
            for c_part in range(num_part):
                accustr += f'\t{c_part}_miou\t{iou_per_part_avg[c_part]}'
            print(accustr)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.cfloat))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    is_show_img = False

    if is_show_img:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)

    args = parse_args()

    main(args)


