import os
import cv2
import time

import random
import pandas as pd
import numpy as np
import math

import timm

from box import Box

import albumentations
from albumentations.pytorch import ToTensorV2

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from apex import amp

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# 获取GPU编号
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
# 当前GPU编号
local_rank = int(FLAGS.local_rank)


# 初试化ddp后端
# torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

# 设置GPU设备
device = torch.device('cuda', local_rank)

'''
train_path = '/data/cheng/experiment/train_images/'
test_path = '/data/cheng/experiment/test_images/'
train_metadata_path = '/data/cheng/experiment/train.csv'
submission_path = '/data/cheng/experiment/sample_submission.csv'
save_path = '/data/cheng/experiment/result/ckpt/'
'''

train_path = '/data/data_ext/zry/happywhale/data/train_images/'
test_path = '/data/data_ext/zry/happywhale/data/test_images/'
train_metadata_path = '/data/data_ext/zry/happywhale/data/train.csv'
submission_path = '/data/data_ext/zry/happywhale/data/sample_submission.csv'
save_path = '/data/data_ext/zry/happywhale/result/'

train_num = len(os.listdir(train_path))
test_num = len(os.listdir(test_path))
train_metadata = pd.read_csv(train_metadata_path)
sample_submission = pd.read_csv(submission_path)

if local_rank == 0:
    print(f'Train Data num is : {train_num} \nTest Data num is : {test_num}')
    print(f'Num_classes is : {train_metadata.individual_id.nunique()}')

# 带有预训练的模型
# model_names = timm.list_models(pretrained=True)
# if local_rank == 0:
#     for model in model_names:
#         if 'eff' in model:
#             print(model)

CONFIG = {
    'RANDOM_SEED': 2022,
    'N_SPLITS': 5,
    'img_size': 448,
    'epochs': 20,
    'experiment_id': 'eff-b0-cutout',
    'embedding_size': 512,
    'train_dataloader': {
        'batch_size': 24,
        'shuffle': True,
        'num_workers': 20,
    },
    'val_dataloader': {
        'batch_size': 24,
        'shuffle': False,
        'num_workers': 20,
    },
    'test_dataloader': {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 20,
    },
    'model': {
        'name': 'tf_efficientnet_b0_ns',
        'output_dim': 15587,
    },
    'arcface_module': {
        # ArcFace Hyperparameters
        "use_arcface": True,
        "s": 30.0,
        "m": 0.50,
        "ls_eps": 0.0,
        "easy_margin": False
    },
    'optimizer': {
        'name': 'optim.Adam',
        'params': {
                'lr': 5e-4,
        }
    },
    # 学习率衰减
    'scheduler': {
        'name': 'optim.lr_scheduler.CosineAnnealingLR',
        'params': {
                'T_max': 500,
                'eta_min': 1e-6,
        }
    },

}
config = Box(CONFIG)

save_location = os.path.join(save_path, config.experiment_id)
if not os.path.exists(save_location):
    os.makedirs(save_location)


def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为单张GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 分布式训练多张GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    # 设置卷积算法固定,每次都使用固定卷积算法,benckmark为True会搜寻最适合的卷积算法,所以deterministic和benckmark是相反的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(config.RANDOM_SEED)

# 修改成26个物种
train_metadata.species.replace({
    "globis": "short_finned_pilot_whale",
    "pilot_whale": "short_finned_pilot_whale",
    "kiler_whale": "killer_whale",
    "bottlenose_dolpin": "bottlenose_dolphin",
}, inplace=True)
if local_rank == 0:
    print('物种数量:', train_metadata.species.nunique())

# labelEncoder的fit统计唯一值，transform转换成0,1...的编码
le = LabelEncoder()
train_metadata['individual_id'] = le.fit_transform(
    train_metadata['individual_id'])


def get_default_transforms(mode):
    if mode == 'train':
        aug = albumentations.Compose([
            albumentations.Resize(config.img_size, config.img_size, p=1),
            albumentations.CoarseDropout(max_holes=20, max_height=20, max_width=20, fill_value=64, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ])
    else:
        aug = albumentations.Compose([
            albumentations.Resize(config.img_size, config.img_size, p=1),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ])

    return aug


# 自定义DataSet
class Train_HappyWhaleDataSet(Dataset):
    def __init__(self, df, transforms=None):
        # df的某一列为series,series的values为numpy.array
        self.img_name = df['image'].values
        self.labels = df['individual_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img = cv2.imread(os.path.join(train_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transforms != None:
            img = self.transforms(image=img)['image']

        return img, label


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + \
                self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def extract_feat(self, input):
        emb = F.linear(input, self.weight)

        return emb

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class HappyWhaleModel(nn.Module):
    def __init__(self, config):
        super(HappyWhaleModel, self).__init__()
        self.cfg = config
        self.backbone = timm.create_model(
            self.cfg.model.name, pretrained=True)
        self.num_features = self.backbone.num_features
        if self.cfg.arcface_module.use_arcface:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
            self.pooling = GeM()
            # self.drop = nn.Dropout(p=0.2, inplace=False)
            self.embedding = nn.Linear(in_features, config.embedding_size)
            self.fc = ArcMarginProduct(config.embedding_size,
                                       self.cfg.model.output_dim,
                                       s=self.cfg.arcface_module.s,
                                       m=self.cfg.arcface_module.m,
                                       easy_margin=self.cfg.arcface_module.easy_margin,
                                       ls_eps=self.cfg.arcface_module.ls_eps)
        else:
            self.fc = nn.Linear(self.num_features, self.cfg.model.output_dim)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.cfg.arcface_module.use_arcface:
            pooled_features = self.pooling(features).flatten(1)
            embedding = self.embedding(pooled_features)
            # pooled_drop = self.drop(x)
            if labels == None:
                x = self.fc.extract_feat(pooled_features)
                return x
            out = self.fc(embedding, labels)
        else:
            out = self.fc(x)

        return out

    def extract_feat(self, x):
        x = self.backbone(x)
        if self.cfg.arcface_module.use_arcface:
            x = self.fc.extract_feat(x)

        return x

def reduce_tensor(tensor, reduction=True):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if reduction:
        rt /= config.int(os.environ["WORLD_SIZE"])
    return rt

# 自定义损失
# 多分类的交叉熵损失，损失函数声明记得加括号
def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

##计算MAP@5ACC
def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])

##收集多卡数据
def gather_tensor(tensor):
    rt = tensor.clone()

    tensor_list = [torch.zeros_like(rt) for _ in range(int(os.environ["WORLD_SIZE"]))]
    dist.all_gather(tensor_list, rt)
    return tensor_list

##计算所有卡的平均值
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def Train(model, train_dataloader, epoch, optimizer, scheduler):
    model.train()
    loss_total = 0
    data_size = 0
    for batch_idx, (imgs, labels) in enumerate(train_dataloader):
        imgs = imgs.cuda(local_rank)
        labels = labels.cuda(local_rank)
        id_preds = model(imgs, labels)
        ce_loss = criterion(id_preds, labels)
        reduced_loss = reduce_mean(ce_loss, int(os.environ["WORLD_SIZE"]))


        id_preds_gather = gather_tensor(id_preds)

        id_preds_gather = torch.cat(id_preds_gather, dim=0)
        # print(id_preds_gather.shape)
        target_list = gather_tensor(labels)
        target_list = torch.cat(target_list, 0)
        _, top5_index = id_preds_gather.topk(5, 1, True, True)

        # 反向传播
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        if batch_idx % 10 == 0 and local_rank == 0:
            thisbatch_acc = map_per_set(target_list.tolist(), top5_index.tolist())
            print(
                f'Epoch [{epoch + 1}] {batch_idx}/{len(train_dataloader)} train Loss : {reduced_loss.item()}  当前batch精度：{thisbatch_acc}')

        # if batch_idx == 1:
        # 	break
        loss_total += reduced_loss.item() * imgs.shape[0] * int(os.environ["WORLD_SIZE"])
        data_size += imgs.shape[0] * int(os.environ["WORLD_SIZE"])


        train_avg_loss = loss_total / data_size


    return model, train_avg_loss


def Val(model, val_dataloader, epoch):
    model.eval()
    loss_total = 0
    acc_total = 0
    data_size = 0
    total_pre =[ ]
    total_label = []
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_dataloader):
            imgs = imgs.cuda(local_rank)
            labels = labels.cuda(local_rank)
            id_preds = model(imgs,labels)


            id_preds_gather = gather_tensor(id_preds)
            id_preds_gather = torch.cat(id_preds_gather, dim=0)
            target_list = gather_tensor(labels)
            target_list = torch.cat(target_list, 0)
            # if local_rank == 0:
            total_pre.append(id_preds_gather)
            total_label.append(target_list)
            # print(id_preds_gather.shape)


            # prob = F.softmax(id_preds, dim=1)
            # _, predicts = torch.max(prob, 1)
            # acc = (predicts == labels).type(
            #         torch.cuda.FloatTensor).sum() * 1.0
            # reduced_acc = reduce_tensor(acc.data,reduction=False)
            # acc_total += reduced_acc.item()

            ce_loss = criterion(id_preds, labels)
            reduced_loss = reduce_mean(ce_loss, int(os.environ["WORLD_SIZE"]))
            loss_total += reduced_loss.item() * imgs.shape[0] * int(os.environ["WORLD_SIZE"])

            data_size += imgs.shape[0] * int(os.environ["WORLD_SIZE"])


        val_avg_loss = loss_total / data_size
        total_pre = torch.cat(total_pre, 0)
        total_label = torch.cat(total_label, 0)
        _, top5_index = total_pre.topk(5, 1, True, True)
        val_avg_acc = map_per_set(total_label.tolist(), top5_index.tolist())

    return val_avg_loss, val_avg_acc


# 五折交叉验证
preds = np.zeros(test_num)
kf = KFold(n_splits=config.N_SPLITS, shuffle=True,
           random_state=config.RANDOM_SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_metadata)):
    # 创建模型
    model = HappyWhaleModel(config)
    # 模型加载到当前GPU上
    model.cuda(device)

    # model.cuda()
    #     test_model = create_model()
    # 声明优化器和学习率变化策略
    optimizer = eval(config.optimizer.name)(
        model.parameters(), lr=config.optimizer.params.lr)
    scheduler = eval(config.scheduler.name)(optimizer, T_max=config.scheduler.params.T_max,
                                            eta_min=config.scheduler.params.eta_min)
    # DDP包裹
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if local_rank == 0:
        print('=' * 25, f'Fold : {fold + 1} training', '=' * 25)
    # train_DataSet
    train_df = train_metadata.loc[train_idx, :]
    train_transforms = get_default_transforms(mode='train')
    train_dataset = Train_HappyWhaleDataSet(
        train_df, transforms=train_transforms)
    # sampler封装dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_dataloader.batch_size,
                                  sampler=train_sampler, num_workers=config.train_dataloader.num_workers)

    # val_DataSet
    val_df = train_metadata.loc[val_idx, :]
    val_transforms = get_default_transforms(mode='val')
    val_dataset = Train_HappyWhaleDataSet(val_df, transforms=val_transforms)
    # sampler封装dataset
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_dataloader.batch_size,
                                sampler=val_sampler, num_workers=config.val_dataloader.num_workers)

    best_loss = np.inf
    time_1 = time.time()
    for epoch in range(config.epochs):
        # train
        model, train_avg_loss = Train(
            model, train_dataloader, epoch, optimizer, scheduler)
        # val
        val_avg_loss, val_avg_acc = Val(model, val_dataloader, epoch)
        if local_rank == 0:
            print('=' * 10, f'Epoch : {epoch + 1} result', '=' * 10)
            print(
                f'Epoch [{epoch + 1}] train_avg_Loss : {train_avg_loss}, val_avg_Loss : {val_avg_loss}, val_avg_acc:{val_avg_acc}')
            if val_avg_loss <= best_loss:
                best_loss = val_avg_loss
                torch.save(model.state_dict(), save_location +
                                                f'/fold_{fold}_best_epoch.pth')
            time_2 = time.time()
            print((time_2 - time_1) / 60)
    if local_rank == 0:
        print('=' * 10, f'Fold : {fold + 1} result', '=' * 10)
        print(f'Best_Loss : {best_loss}')
    break

