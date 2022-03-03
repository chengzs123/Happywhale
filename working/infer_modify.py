import os
import cv2
import time

import random
import pandas as pd
import numpy as np
import math

import timm

from tqdm import tqdm

import faiss

import joblib

from box import Box

import albumentations
from albumentations.pytorch import ToTensorV2

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt

from collections import OrderedDict

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, normalize

# 声明数据路径，读取数据
train_path = '/data/cheng/experiment/train_images/'
test_path = '/data/cheng/experiment/test_images/'
train_metadata_path = '/data/cheng/experiment/train.csv'
submission_path = '/data/cheng/experiment/sample_submission.csv'
save_path = '/data/cheng/experiment/result/ckpt/'

train_num = len(os.listdir(train_path))
test_num = len(os.listdir(test_path))
train_metadata = pd.read_csv(train_metadata_path)
sample_submission = pd.read_csv(submission_path)

print(f'Train Data num is : {train_num} \nTest Data num is : {test_num}')
print(f'Num_classes is : {train_metadata.individual_id.nunique()}')

# 带有预训练的模型
# model_names = timm.list_models(pretrained=True)
# if local_rank == 0:
#     for model in model_names:
#         if 'eff' in model:
#             print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# config参数
CONFIG = {
    'RANDOM_SEED': 2022,
    'N_SPLITS': 5,
    'img_size': 448,
    'epochs': 20,
    'experiment_id': 'eff-b0-test',
    'embedding_size': 512,
    'train_dataloader': {
        'batch_size': 96,
        'shuffle': True,
        'num_workers': 20,
    },
    'val_dataloader': {
        'batch_size': 96,
        'shuffle': False,
        'num_workers': 20,
    },
    'test_dataloader': {
        'batch_size': 96,
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

model_save_location = os.path.join(save_path, 'ckpt', config.experiment_id)
if not os.path.exists(model_save_location):
    os.makedirs(model_save_location)

# 设置随机种子固定所有随机性
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
print('物种数量:', train_metadata.species.nunique())

# labelEncoder的fit统计唯一值，transform转换成0,1...的编码
# 这里转换成读取训练时保存好的label_encoder
# le = LabelEncoder()
# train_metadata['individual_id'] = le.fit_transform(
#     train_metadata['individual_id'])
# with open(f'le.pkl', 'wb') as fp:
#     joblib.dump(le, fp)

with open(f'le.pkl', 'rb') as fp:
    le = joblib.load(fp)
train_metadata['individual_id'] = le.fit_transform(
    train_metadata['individual_id'])

# 数据增强
def get_default_transforms(mode):
    if mode == 'train':
        aug = albumentations.Compose([
            albumentations.Resize(config.img_size, config.img_size, p=1),
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


# DataSet
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

        return img, label, img_name

class Test_HappyWhaleDataSet(Dataset):
    def __init__(self, df, transforms=None):
        # df的某一列为series,series的values为numpy.array
        self.img_name = df['image'].values
        self.labels = df['individual_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img = cv2.imread(os.path.join(test_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transforms != None:
            img = self.transforms(image=img)['image']

        return img, label, img_name


# Arcface结构
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
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + \
                self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# GeM池化策略
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

# 模型
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
            self.embedding = nn.Linear(in_features, config.embedding_size)
            self.fc = ArcMarginProduct(config.embedding_size,
                                       self.cfg.model.output_dim,
                                       s=self.cfg.arcface_module.s,
                                       m=self.cfg.arcface_module.m,
                                       easy_margin=self.cfg.arcface_module.easy_margin,
                                       ls_eps=self.cfg.arcface_module.ls_eps)

    # 前向传播得到改善后的标签
    def forward(self, x, labels):
        features = self.backbone(x)
        if self.cfg.arcface_module.use_arcface:
            pooled_features = self.pooling(features).flatten(1)
            embedding = self.embedding(pooled_features)
            out = self.fc(embedding, labels)

        return out

    # 提取嵌入特征
    def extract_feat(self, x):
        features = self.backbone(x)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)

        return embedding

# 获取嵌入特征
def get_embedding(model, data_loader):
    model.eval()
    labels_total = []
    embeddings_total = []
    img_names_total = []
    # bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for i, (imgs, labels, img_names) in enumerate(data_loader):
            if i % 10 == 0:
                print(f'提取嵌入特征:{i}/{len(data_loader)}')
            imgs = imgs.cuda()
            labels = labels.cuda()
            embedding = model.extract_feat(imgs)

            labels_total.append(labels.cpu().data.numpy())
            embeddings_total.append(embedding.cpu().data.numpy())
            img_names_total.append(img_names)

    embeddings_total = np.vstack(embeddings_total)
    labels_total = np.concatenate(labels_total)
    img_names_total = np.concatenate(img_names_total)

    return labels_total, embeddings_total, img_names_total

# 加载模型
def load_model(ckpt_path, config):
    model = HappyWhaleModel(config)
    state_dict = torch.load(ckpt_path,
                            map_location=torch.device('cpu'))
    new_s = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_s[name] = v
    model.load_state_dict(new_s)
    model.cuda()

    return model

# 利用faiss获取KNN，通过内积计算获取余弦相似性
def KNN(embedding_1, embedding_2):
    '''
    :param embedding_1:进行内积的矩阵
    :param embedding_2:进行查找的矩阵
    :return:D:余弦距离矩阵 I:索引矩阵
    '''
    # 本身除以L2范数
    embedding_1 = normalize(embedding_1, axis=1, norm='l2')
    embedding_2 = normalize(embedding_2, axis=1, norm='l2')

    # 基于余弦相似度进行50近邻查找
    index = faiss.IndexFlatIP(config.embedding_size)
    index.add(embedding_1)
    D, I = index.search(embedding_2, 50)

    return D, I

# 预测
def get_predictions(test_df, threshold=0.2):
    predictions = {}
    # row三个数据，target，distance和image
    for i, row in test_df.iterrows():
        # 预测为前五个距离最近的Id
        if row.image in predictions:
            if len(predictions[row.image]) == 5:
                continue
            predictions[row.image].append(row.target)
        # 根据阈值决定将new_individual放在第一位还是第二位
        elif row.distances > threshold:
            predictions[row.image] = [row.target, 'new_individual']
        else:
            predictions[row.image] = ['new_individual', row.target]

    # for x in tqdm(predictions):
    #     if len(predictions[x]) < 5:
    #         remaining = [y for y in sample_list if y not in predictions]
    #         predictions[x] = predictions[x] + remaining
    #         predictions[x] = predictions[x][:5]

    return predictions

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

# test_df
test = pd.DataFrame()
test["image"] = os.listdir("/data/cheng/experiment/test_images")
test["individual_id"] = -1  #dummy value
print(test.head())
# 声明test_dataset
test_transform = get_default_transforms(mode='test')
test_dataset = Test_HappyWhaleDataSet(test, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=config.test_dataloader.batch_size,
                             num_workers=config.test_dataloader.num_workers, shuffle=False)

# 交叉验证设置
kf = KFold(n_splits=config.N_SPLITS, shuffle=True,
           random_state=config.RANDOM_SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_metadata)):
    model = load_model('/data/cheng/experiment/result/ckpt/eff-b0-test/fold_0_best_epoch.pth', config)
    # train_DataSet
    train_df = train_metadata.loc[train_idx, :]
    # 这里相当于推理提取嵌入，使用和val以及test相同的transform
    train_transforms = get_default_transforms(mode='val')
    train_dataset = Train_HappyWhaleDataSet(train_df, transforms=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_dataloader.batch_size,
                                num_workers=config.train_dataloader.num_workers, shuffle=True)

    # val_DataSet
    val_df = train_metadata.loc[val_idx, :]
    val_transforms = get_default_transforms(mode='val')
    val_dataset = Train_HappyWhaleDataSet(val_df, transforms=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_dataloader.batch_size,
                                shuffle=True, num_workers=config.val_dataloader.num_workers)

    # 获取label和train嵌入特征
    labels_train, embedding_train, img_names_train = get_embedding(model, train_dataloader)
    labels_val, embedding_val, img_names_val = get_embedding(model, val_dataloader)

    # 标签转换成ID
    labels_train = le.inverse_transform(labels_train)
    labels_val = le.inverse_transform(labels_val)

    # 通过CV计算新id的阈值
    D_1, I_1 = KNN(embedding_train, embedding_val)
    # val中出现但是train中没出现的声明为新ID
    allowed_targets = np.unique(labels_train)
    val_targets_df = pd.DataFrame(np.stack([img_names_val, labels_val], axis=1), columns=['image', 'target'])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), 'target'] = 'new_individual'
    # 记录对于val的预测的k近邻ID和距离
    valid_df = []
    for i, val_id in enumerate(img_names_val):
        targets = labels_train[I_1[i]]
        distances = D_1[i]
        subset_preds = pd.DataFrame(np.stack([targets, distances], axis=1), columns=['target', 'distances'])
        subset_preds['image'] = val_id
        valid_df.append(subset_preds)
    # 对于相同的ID的距离取最大的一个
    valid_df = pd.concat(valid_df).reset_index(drop=True)
    valid_df = valid_df.groupby(['image', 'target']).distances.max().reset_index()
    # 通过CV值选取最好的threshhold值
    best_th = 0
    best_cv = 0
    for th in [0.1 * x for x in range(11)]:
        all_preds = get_predictions(valid_df, threshold=th)
        cv = 0
        for i, row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i, th] = map_per_image(target, preds)
        cv = val_targets_df[th].mean()
        print(f"CV at threshold {th}: {cv}")
        if cv > best_cv:
            best_th = th
            best_cv = cv
    val_targets_df['is_new_individual'] = val_targets_df.target == 'new_individual'
    print(val_targets_df.is_new_individual.value_counts().to_dict())
    print(val_targets_df.head())
    val_scores = val_targets_df.groupby('is_new_individual').mean().T
    val_scores['adjusted_cv'] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_threshold_adjusted = val_scores['adjusted_cv'].idxmax()
    print(val_scores)

    # 合并所有训练集
    embedding_train = np.concatenate([embedding_train, embedding_val], axis=0)
    labels_train = np.concatenate([labels_train, labels_val], axis=0)
    print(embedding_train.shape, embedding_val.shape)

    # 获取test的嵌入特征
    _, embedding_test, img_names_test = get_embedding(model, test_dataloader)
    # 计算K近邻
    D, I = KNN(embedding_train, embedding_test)
    # 记录每个test图像的K近邻对应的距离以及对应train中的ID
    test_df = []
    for i, img_name in enumerate(img_names_test):
        targets = labels_train[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([targets, distances], axis=1), columns=['target', 'distances'])
        subset_preds['image'] = img_name
        test_df.append(subset_preds)
    # 因为图像中存在多个图像对应一个id的情况，所以嵌入的k近邻也会出现相同的ID，对于这些相同的ID
    # 取相似度最大(距离最大)的嵌入，然后再对这些相似度做排序
    test_df = pd.concat(test_df).reset_index(drop=True)
    test_df = test_df.groupby(['image', 'target']).distances.max().reset_index()
    test_df = test_df.sort_values('distances', ascending=False).reset_index(drop=True)

    # 预测，最好的阈值根据当前折的CV值确定
    predictions = get_predictions(test_df, best_threshold_adjusted)

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image', 'predictions']
    predictions['predictions'] = predictions['predictions'].apply(lambda x: ' '.join(x))
    predictions.to_csv('submission.csv', index=False)
    predictions.head()

    break
