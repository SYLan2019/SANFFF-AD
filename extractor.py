import numpy as np
import torch
from tqdm import tqdm
import config as c
from model import ADwithGlow
from utils import *
import os
import modeling_finetune


# 我们提取的是处理好的数据所以不用在处理
def extract_train(train_loader, test_loader, class_name):
    model = ADwithGlow()
    model.to(c.device)
    model.eval()
    with torch.no_grad():
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = list()
            labels = list()
            for i, data in enumerate(tqdm(loader)):
                inputs, l = preprocess_batch(data)
                labels.append(t2np(l))
                if c.extractor == 'clip':
                    z = model.feature_extractor(inputs)
                elif c.extractor == 'VIT':
                    z = model.feature_extractor(inputs)
                    z = model.vit_ext(inputs)
                elif c.extractor == 'DINOV2':
                    z = model.dinov2_ext(inputs)
                elif c.extractor == 'DINO':
                    z = model.feature_extractor
                features.append(t2np(z))
            f = np.concatenate(features, axis=0)
            if name == 'test':
                np.save(export_test_dir + 'testfeatures', f)
                labels = np.concatenate(labels)
                np.save(export_test_dir + 'labels', labels)
                print(f.shape)
            else:
                np.save(export_dir + class_name + '_' + name, f)
            print(f.shape)


# class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# class_name = ['airplane']
# class_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
# class_name = [x for x in range(10)]
# class_name = ['Cat','Dog']
class_name = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'LICENSE-DATASET', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
# class_name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
#               'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# class_name = ['lbot_data']
os.environ['TORCH_HOME'] = 'models\\EfficientNet'
# os.environ['TORCH_HOME'] = r'models/EfficientNet'
for i in class_name:
    c.class_name = i
    export_name = str(c.class_name)
    export_dir = 'data/all_features/' + c.extractor + '/' + c.dataset + '/' + 'train' + '/' + export_name + '/'
    export_test_dir = 'data/all_features/' + c.extractor + '/' + c.dataset + '/' + 'test' + '/' + export_name + '/'
    c.pre_extracted = False
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(export_test_dir, exist_ok=True)
    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)
    # train_loader,test_loader = get_loaders(c.dataset,c.class_name,c.batch_size)
    extract_train(train_loader, test_loader, str(c.class_name))
    # extract_test(test_loader)
