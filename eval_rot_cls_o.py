"""
其他方案的旋转训练
训练不变，测试集旋转
"""

# 工具包
import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
from tqdm import tqdm
from colorama import Fore, Back, init
import os

# 自建模块
from data_utils.ParamDataLoader import all_metric_cls, MCBDataLoader
from models.GCN3D import GCN3D
from models.dgcnn import DGCNN
from models.PointNet import PointNet
from models.PointConv import PointConv
from models.PointCNN import PointCNN
from models.PointNet2 import PointNet2
from models.PPFNet import PPFNet


def parse_args():
    '''PARAMETERS'''
    # 输入参数如下：
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=50, help='batch size in training')
    parser.add_argument('--epoch', default=35, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--is_load_weight', type=str, default='False', choices=['True', 'False'], help='---')
    parser.add_argument('--num_point', type=int, default=2000, help='Point Number') # 点数量

    parser.add_argument('--save_str', type=str, default='sdgraph', help='---')
    parser.add_argument('--rotate', default=0, type=float, help='Degree of rotate angle, not radius')
    parser.add_argument('--model', type=str, default='GCN3D', choices=['GCN3D', 'DGCNN', 'PointNet', 'PointNet2', 'PointConv', 'PointCNN', 'PPFNet'], help='model used for cls')

    parser.add_argument('--is_train', default='True', choices=['True', 'False'], type=str, help='training the model?')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='runing on local?')
    parser.add_argument('--root_sever', type=str, default=r'/root/my_data/data_set/STEP20000_Hammersley_2000', help='---')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\STEP20000_Hammersley_2000', help='---')

    # rf'/root/my_data/data_set/unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}'
    # rf'D:\document\DeepLearning\DataSet\unified_sketch_cad_stk{global_defs.n_stk}_stkpnt{global_defs.n_stk_pnt}'
    # r'/root/my_data/data_set/TU_Berlin_std_cls'
    # r'D:\document\DeepLearning\DataSet\TU_Berlin_std_cls'

    return parser.parse_args()


def cls_eval(model, data_loader):
    with torch.no_grad():
        model = model.eval()

        all_preds = []
        all_labels = []

        for j, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            points, target = data[0].float().cuda(), data[1].long().cuda()

            points = points.permute(0, 2, 1)
            assert points.size()[1] == 3

            pred = model(points)

            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())

    return all_preds, all_labels


def main(args):
    save_str = args.save_str
    print(Fore.BLACK + Back.BLUE + 'save as: ' + save_str)

    '''创建文件夹'''
    confusion_dir = save_str + '-' + datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    confusion_dir = os.path.join('data_utils', 'confusion', confusion_dir)
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    model_savepth = 'model_trained/' + save_str + '.pth'
    os.makedirs('log/', exist_ok=True)

    '''日志记录'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''定义数据集'''
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever
    train_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, is_train=True, data_augmentation=False)
    test_dataset = MCBDataLoader(root=data_root, npoints=args.num_point, is_train=False, data_augmentation=False, rotate=args.rotate)
    num_class = len(train_dataset.classes)

    # sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=64, replacement=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    # sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=64, replacement=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    '''加载模型及权重'''
    if args.model == 'GCN3D':
        classifier = GCN3D(support_num=1, neighbor_num=20, classes_num=num_class).cuda()
    elif args.model == 'DGCNN':
        classifier = DGCNN(output_channels=num_class).cuda()
    elif args.model == 'PointNet':
        classifier = PointNet(k=num_class).cuda()
    elif args.model == 'PointNet2':
        classifier = PointNet2(num_class=num_class).cuda()
    elif args.model == 'PointConv':
        classifier = PointConv(num_classes=num_class).cuda()
    elif args.model == 'PointCNN':
        classifier = PointCNN(n_classes=num_class).cuda()
    elif args.model == 'PPFNet':
        classifier = PPFNet(n_classes=num_class).cuda()
    else:
        raise TypeError('error model name!')

    if args.is_load_weight == 'True':
        try:
            classifier.load_state_dict(torch.load(model_savepth))
            print(Fore.GREEN + 'training from exist model: ' + model_savepth)
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''训练'''
    if args.is_train == 'True':
        is_train = True
        print(Fore.BLACK + Back.BLUE + 'train model')
    else:
        is_train = False
        print(Fore.BLACK + Back.BLUE + 'evaluating model only')

    best_instance_accu = -1.0
    for epoch in range(args.epoch):
        if is_train:
            classifier = classifier.train()

            logstr_epoch = f'Epoch({epoch}/{args.epoch}):'
            all_preds = []
            all_labels = []

            for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                points, target = data[0].float().cuda(), data[1].long().cuda()

                # -> [bs, 3, n_points]
                points = points.permute(0, 2, 1)
                assert points.size()[1] == 3

                # 梯度置为零，否则梯度会累加
                optimizer.zero_grad()

                pred = classifier(points)
                loss = F.nll_loss(pred, target)
                # loss = loss_func(pred[0], target, pred[1])
                # pred = pred[0]

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
        all_preds, all_labels = cls_eval(classifier, test_dataloader)
        all_metric_eval = all_metric_cls(all_preds, all_labels, os.path.join(confusion_dir, f'eval-{epoch}.png'))
        accustr = f'\teval_ins_acc\t{all_metric_eval[0]}\teval_cls_acc\t{all_metric_eval[1]}\teval_f1_m\t{all_metric_eval[2]}\teval_f1_w\t{all_metric_eval[3]}\tmAP\t{all_metric_eval[4]}'

        if is_train:
            logger.info(logstr_epoch + logstr_trainaccu + accustr)
            print(f'{save_str}: epoch {epoch}/{args.epoch}: train_ins_acc: {all_metric_train[0]}, test_ins_acc: {all_metric_eval[0]}')
        else:
            print(f'{save_str}: test_ins_acc: {all_metric_eval[0]}')
            exit(0)

        # 额外保存最好的模型
        if best_instance_accu < all_metric_eval[0]:
            best_instance_accu = all_metric_eval[0]
            torch.save(classifier.state_dict(), 'model_trained/best_' + save_str + '.pth')


if __name__ == '__main__':
    # clear_log('./log')
    # clear_confusion('./data_utils/confusion')
    init(autoreset=True)
    main(parse_args())
























