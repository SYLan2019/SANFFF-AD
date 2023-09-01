import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

import config as c
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from PIL import Image
from torchvision.datasets import CIFAR10, STL10, FashionMNIST
import faiss
import csv
import random


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    z = z.reshape(z.shape[0], -1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


transform_color = transforms.Compose([transforms.Resize(c.img_size),
                                      transforms.CenterCrop(c.img_size),
                                      transforms.ToTensor(),
                                      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
transform_clip = transforms.Compose([transforms.Resize(c.img_size),
                                     transforms.CenterCrop(c.img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711])
                                     ])


def get_test_transforms(input_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transformations = transforms.Compose(
        [transforms.Resize(input_size, interpolation=3),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    return transformations


transform_gray = transforms.Compose([
    transforms.Resize(c.img_size),
    transforms.CenterCrop(c.img_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


# 队训练集和测试集进行随机打乱
def syn_shuffle(lst0, lst1, lst2, lst3):
    lst = list(zip(lst0, lst1, lst2, lst3))
    random.shuffle(lst)
    lst0, lst1, lst2, lst3 = zip(*lst)
    return lst0, lst1, lst2, lst3


class VisaDataset(Dataset):

    def __init__(self, root, transform, gt_transform, phase, category=None, split_ratio=1):
        self.phase = phase
        self.root = root
        self.category = category
        self.transform = transform
        self.gt_transform = gt_transform
        self.split_ratio = split_ratio
        self.split_file = root + "/split_csv/1cls.csv"
        assert os.path.isfile(self.split_file), 'Error VsiA dataset'
        assert os.path.isdir(os.path.join(self.root, category)), 'Error VsiA dataset category:{}'.format(category)

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        with open(self.split_file, 'r') as file:
            csvreader = csv.reader(file)
            # 第一行读掉
            next(csvreader)
            for row in csvreader:
                category, split, label, image_path, mask_path = row
                image_name = image_path.split("/")[-1]
                mask_name = mask_path.split("/")[-1]
                if split == 'train' and self.phase == 'eval':
                    split = 'eval'
                # 符合要求的category
                if self.phase == split and self.category == category:
                    img_src_path = os.path.join(self.root, image_path)
                    if label == "normal":
                        gt_src_path = 0
                        index = 0
                        types = "good"
                    else:
                        index = 1
                        types = "bad"
                        gt_src_path = os.path.join(self.root, mask_path)

                    img_tot_paths.append(img_src_path)
                    gt_tot_paths.append(gt_src_path)
                    tot_labels.append(index)
                    tot_types.append(types)
        train_len = int(len(img_tot_paths) * self.split_ratio)
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = syn_shuffle(img_tot_paths, gt_tot_paths, tot_labels,
                                                                         tot_types)
        if self.phase == "train":
            img_tot_paths = img_tot_paths[:train_len]
            gt_tot_paths = gt_tot_paths[:train_len]
            tot_labels = tot_labels[:train_len]
            tot_types = tot_types[:train_len]
        elif self.phase == 'eval':
            img_tot_paths = img_tot_paths[train_len:]
            gt_tot_paths = gt_tot_paths[train_len:]
            tot_labels = tot_labels[train_len:]
            tot_types = tot_types[train_len:]

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        origin = img
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            if(self.gt_transform!=None):
                gt = self.gt_transform(gt)

        # assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, label
        # return img, gt, label, os.path.basename(img_path[:-4]), img_type
        # return {
        #     'origin': np.array(origin),
        #     'image': img,
        #     'gt': gt,
        #     'label': label,
        #     'name': os.path.basename(img_path[:-4]),
        #     'type': img_type
        # }


# sensory dataset
def load_datasets(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    if c.dataset == 'mvtec':
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test')

        classes = os.listdir(data_dir_test)
        if 'good' not in classes and c.dataset == 'mvtec':
            print(
                'There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if cl == 'good':
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1

        # tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]
        # transform_train = transforms.Compose(tfs)

        tfs = [transforms.Resize(c.img_size, Image.ANTIALIAS),
               transforms.CenterCrop(c.img_size),
               transforms.ToTensor(),
               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
               ]
        tfs_test = [transforms.Resize(c.img_size, Image.ANTIALIAS),
                    transforms.CenterCrop(c.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        if c.class_name in ['zipper', 'screw', 'grid']:
            tfs = [transforms.Resize(c.img_size),
                   transforms.CenterCrop(c.img_size),
                   transforms.Grayscale(num_output_channels=3),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            tfs_test = [transforms.Resize(c.img_size, Image.ANTIALIAS),
                        transforms.CenterCrop(c.img_size),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transform_train = transforms.Compose(tfs)
        transform_test = transforms.Compose(tfs_test)
        trainset = ImageFolder(data_dir_train, transform=transform_train)
        testset = ImageFolder(data_dir_test, transform=transform_test, target_transform=target_transform)

    elif c.dataset == 'visa':
        trainset = VisaDataset(dataset_path, transform=transform_color, gt_transform=None, phase="train",
                               category=class_name, split_ratio=1)
        testset = VisaDataset(dataset_path, transform=transform_color, gt_transform=None, phase="test",
                              category=class_name, split_ratio=1)
    elif c.dataset == 'btad':
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test')
        classes = os.listdir(data_dir_test)
        if 'good' not in classes and c.dataset == 'btad':
            print(
                'There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if cl == 'good':
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1
        trainset = ImageFolder(data_dir_train, transform=transform_color)
        testset = ImageFolder(data_dir_test, transform=transform_color,target_transform=target_transform)
    elif c.dataset == 'CatsvsDogs':
        data_dir_train = os.path.join(dataset_path, 'Train', 'Train_' + class_name)
        data_dir_test = dataset_path + 'Test'
        classes = os.listdir(data_dir_test)
        class_perm = list()
        for cl in classes:
            if cl == class_name:
                class_perm.append(0)
            else:
                class_perm.append(1)
        trainset = ImageFolder(data_dir_train, transform=transform_color)
        testset = ImageFolder(data_dir_test, transform=transform_color, target_transform=target_transform)

    elif c.dataset == 'lbot':
        root = dataset_path  # 'data1/lbot
        train_dataset = LBOT_Dataset(root, c.img_size, transform=transform_color)
        test_dataset = LBOT_Dataset(root, c.img_size, transform=transform_color, istrain=False)
        return train_dataset, test_dataset
    else:
        raise AttributeError
    return trainset, testset


def make_dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


# sematic dataset
def get_loaders(dataset, label_class, batch_size):
    if c.pretrained:
        trainset = FeatureDataset(train=True)
        testset = FeatureDataset(train=False)
        # label_class = trainset.class_to_idx[label_class]
    else:
        if dataset in ['cifar10', 'fashion', 'dior']:
            if dataset == "cifar10":
                ds = torchvision.datasets.CIFAR10
                transform = get_test_transforms(c.img_size) if c.extractor != 'clip' else transform_clip

                coarse = {}
                trainset = ds(root='data1/cifar10', train=True, download=True, transform=transform, **coarse)
                testset = ds(root='data1/cifar10', train=False, download=True, transform=transform, **coarse)
                label_class = trainset.class_to_idx[label_class]
            elif dataset == "fashion":
                ds = torchvision.datasets.FashionMNIST
                transform = transform_gray
                coarse = {}
                trainset = ds(root='data1/FashionMNIST', train=True, download=True, transform=transform, **coarse)
                testset = ds(root='data1/FashionMNIST', train=False, download=True, transform=transform, **coarse)

            idx = np.array(trainset.targets) == int(label_class)
            testset.targets = [int(t != label_class) for t in testset.targets]
            trainset.data = trainset.data[idx]
            trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        elif dataset in ['STL10']:
            ds = torchvision.datasets.STL10
            transform = transform_color
            trainset = ds(root='data1/STL10', split='train', download=True, transform=transform)
            testset = ds(root='data1/STL10', split='test', download=True, transform=transform)
            label_class = trainset.classes.index(label_class)
            idx = np.array(trainset.labels) == label_class
            testset.labels = [int(t != label_class) for t in testset.labels]
            trainset.data = trainset.data[idx]
            trainset.labels = [trainset.labels[i] for i, flag in enumerate(idx, 0) if flag]
        elif dataset in ['CIFAR100']:
            ds = torchvision.datasets.CIFAR100
            transform = transform_color
            testset = ds(root='data1/CIFAR100',
                         train=False, download=True,
                         transform=transform)

            trainset = ds(root='data1/CIFAR100',
                          train=True, download=True,
                          transform=transform)

            trainset.targets = sparse2coarse(trainset.targets)
            testset.targets = sparse2coarse(testset.targets)
            # 暂时设置一下
            # label_class = trainset.classes.index(label_class)
            idx = np.array(trainset.targets) == label_class
            testset.targets = [int(t != label_class) for t in testset.targets]
            trainset.data = trainset.data[idx]
            trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


# def knn_score(train_set, test_set, n_neighbours=5):
#     """
#     Calculates the KNN distance
#     """
#     index = faiss.IndexFlatL2(train_set.shape[1])
#     index.add(train_set)
#     D, _ = index.search(test_set, n_neighbours)
#     # 计算相似度，D返回的是test_set与train_set中心的近邻居距离大小
#     return np.sum(D, axis=1)


class FeatureDataset(Dataset):
    def __init__(self, root="./data/features/", train=False):

        super(FeatureDataset, self).__init__()
        self.train = train
        suffix = 'train' if train else 'test'
        if train:
            root = root + c.extractor + '/' + c.dataset + '/' + suffix + '/' + c.class_name + '/'
            self.data = np.load(root + c.class_name + '_' + suffix + '.npy')
        else:
            root = root + c.extractor + '/' + c.dataset + '/' + suffix + '/' + c.class_name + '/'
            self.data = np.load(root + 'testfeatures' + '.npy')
        self.labels = np.load(os.path.join(root, 'labels.npy')) if not train else np.zeros(
            [len(self.data)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data
        sample = d[index]
        sample = torch.FloatTensor(sample)
        out = sample
        return out, self.labels[index]


# data to (device)
def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    # inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# 将图片打开成accimage类型的
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, is_train):
    pairs = list()
    if is_train:
        images = glob(os.path.join(dir, 'good', '*.png'))
        if len(images) == 0:
            images = glob(os.path.join(dir, 'train', '0.normal', '*.jpg'))
        # print(glob(os.path.join(dir,'good')))
    else:
        images = glob(os.path.join(dir, '*', '*.png'))
        if len(images) == 0:
            images = glob(os.path.join(dir, 'test', '*', '*.jpg'))
    for i in images:
        if os.path.dirname(i).endswith('abnormal'):
            item = (i, 1)
        elif os.path.dirname(i).endswith('good') or os.path.dirname(i).endswith('normal'):
            item = (i, 0)
        pairs.append(item)
    return pairs


class LBOT_Dataset(Dataset):
    def __init__(self, root, input_size, transform=None, loader=default_loader, istrain=True):
        self.root = root
        self.is_size = input_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.is_size),
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            ])
        else:
            self.transform = transform
        self.imgs = make_dataset(self.root, istrain)
        self.loader = loader

    def __getitem__(self, index):
        path, lable = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, lable

    def __len__(self):
        return len(self.imgs)


# train_set,test_set = load_datasets(dataset_path='data1/archive',class_name='flowers')
if __name__ == '__main__':
    # trainset, testset = load_datasets(dataset_path='./data/visa', class_name='candle')
    trainset, testset = load_datasets(dataset_path='./data', class_name='bottle')
    trainloader, testloader = make_dataloaders(trainset, testset)
    for i, data in enumerate(tqdm(testloader)):
        inputs, l = preprocess_batch(data)
        print(1)