import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config as c
# from localization import export_gradient_maps
from model import ADwithGlow, save_model, save_weights
from utils import *
from evaluate import compare_histogram


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, is_anomaly, anomaly_score, epoch, print_score=False, train_auc=None):
        score = roc_auc_score(is_anomaly, anomaly_score)
        train_auc.append(score)
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            # export_dir = 'data/z_features/' + c.dataset + '/' + 'train' + '/' + c.class_name + '/'
            # export_test_dir = 'data/z_features/' + c.dataset + '/' + 'test' + '/' + c.class_name + '/'
            # np.save(export_dir + c.class_name + '_' + 'train', train_features)
            # np.save(export_test_dir + 'testfeatures', test_features)
            # np.save(export_test_dir + 'labels', is_anomaly)
            # os.makedirs(export_dir, exist_ok=True)
            # os.makedirs(export_test_dir, exist_ok=True)
            self.figure(is_anomaly, anomaly_score, c.class_name)
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))

    def figure(self, is_anomaly, anomaly_score, class_name):
        compare_histogram(scores=anomaly_score, classes=is_anomaly, bins1=c.nbins1, bins2=c.nbins2,
                          class_name=class_name, )


# def train(train_loader, test_loader):
#     model = ADwithGlow()
#     optimizer = torch.optim.AdamW(model.nf_mlp.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04,
#                                  weight_decay=1e-5)
#
#     optimizer_fea = torch.optim.Adam(model.feature_aggreator.parameters(),lr=1e-5)
#     model.to(c.device)
#
#     score_obs = Score_Observer('AUROC')
#     print(F'\nTrain on {c.class_name}')
#     for epoch in range(c.meta_epochs):
#         # train some epochs
#         train_feature_space = list()
#         model.train()
#         if c.verbose:
#             print(F'\nTrain epoch {epoch}')
#         for sub_epoch in range(c.sub_epochs):
#             train_loss = list()
#             for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
#                 optimizer.zero_grad()
#                 optimizer_fea.zero_grad()
#                 # 一类和纹理是不一样的
#                 inputs, labels = preprocess_batch(data)  # move to device and reshape
#                 # train_feature_space.append(inputs)
#                 z = model.feature_aggreator(inputs)
#                 z = model(z)
#                 train_feature_space.append(z)
#                 loss = get_loss(z, model.nf_mlp.jacobian(run_forward=False))
#                 train_loss.append(t2np(loss))
#                 loss.backward()
#                 optimizer.step()
#                 optimizer_fea.step()
#             mean_train_loss = np.mean(train_loss)
#             if c.verbose:
#                 print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
#         # evaluate
#         model.eval()
#         if c.verbose:
#             print('\nCompute loss and scores on test set:')
#         test_loss = list()
#         test_z = list()
#         test_labels = list()
#         with torch.no_grad():
#             for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
#                 inputs, labels = preprocess_batch(data)
#                 # test_feature_space.append(inputs)
#                 z = model.feature_aggreator(inputs)
#                 z = model(z)
#                 # z = model(inputs)
#                 loss = get_loss(z, model.nf_mlp.jacobian(run_forward=False))
#                 test_z.append(z)
#                 test_loss.append(t2np(loss))
#                 test_labels.append(t2np(labels))
#         test_loss = np.mean(np.array(test_loss))
#         if c.verbose:
#             print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))
#
#         test_labels = np.concatenate(test_labels)
#         is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
#         z_grouped = torch.cat(test_z, dim=0)
#         train_features = torch.cat(train_feature_space, dim=0).contiguous().cpu().detach().numpy()
#         test_features = z_grouped.contiguous().cpu().detach().numpy()
#         # (b,c)  (b,c,h,w)
#         z_grouped = z_grouped.reshape(z_grouped.shape[0], -1)
#         anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=1))
#         score_obs.update(is_anomaly, anomaly_score, epoch, train_features, test_features,
#                          print_score=c.verbose or epoch == c.meta_epochs - 1)
#
#     if c.save_model:
#         model.to('cpu')
#         save_model(model, c.modelname)
#         save_weights(model, c.modelname)
#     return model

def train(train_loader, test_loader):
    model = ADwithGlow()
    optimizer = torch.optim.Adam(model.nf_mlp.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04,
                                 weight_decay=1e-5)
    model.to(c.device)

    score_obs = Score_Observer('AUROC')
    train_loss_save = list()
    train_auc = list()
    print(F'\nTrain on {c.class_name}')
    for epoch in range(c.meta_epochs):
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                # optimizer.zero_grad()
                # 一类和纹理是不一样的
                inputs, labels = preprocess_batch(data)  # move to device and reshape
                z = model(inputs)
                loss = get_loss(z, model.nf_mlp.jacobian(run_forward=False))
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()
                break
            mean_train_loss = np.mean(train_loss)
            train_loss_save.append(mean_train_loss)
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                z = model(inputs)
                loss = get_loss(z, model.nf_mlp.jacobian(run_forward=False))
                test_z.append(z)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))
                break
        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
        z_grouped = torch.cat(test_z, dim=0)
        # test_features = z_grouped.contiguous().cpu().detach().numpy()
        # (b,c)  (b,c,h,w)
        z_grouped = z_grouped.reshape(z_grouped.shape[0], -1)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=1))
        score_obs.update(is_anomaly, anomaly_score, epoch,
                         print_score=c.verbose or epoch == c.meta_epochs - 1, train_auc=train_auc)
    with open("./train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss_save))
        # distances = knn_score(train_features, test_features)
        # auc = roc_auc_score(is_anomaly, distances)
        # print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)
    return model
