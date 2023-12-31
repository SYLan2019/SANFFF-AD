'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''
# from freia_funcs import *
# device settings
device = 'cpu'  # or 'cpu'

# data settings
dataset_path = "./data/mvtec"  # parent directory of datasets
dataset = 'mvtec' # 'cifar10'|'STL10|CIFAR100|fashion|CatsvsDogs|mvtec|lbot|visa|btad'-------mvtec:./data|lbot:./data/lbot
class_name = "Dog"  # dataset subdirectory
modelname = "dummy_test"  # export evaluations/logs with this name
subnet = None
img_size = (384,384)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
clamp = 2  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 4  # higher = more flexible = more unstable
fc_internal =1024   # * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 5e-5 # inital learning rate
use_gamma = True
pretrained = True

extractor = "VIT"  # feature dataset name (which was used in 'extract_features.py' as 'export_name')
n_feat = {"clip": 1024,'DINO':768,'VIT':1024,'deit':768,'DINOV2':1024}[extractor]  # dependend from feature extractor

if extractor in ['VIT','deit']:
    img_size = (384,384)
elif extractor in ['dino','ITPN']:
    img_size = (224,224)
elif extractor in ['clip','DINOV2']:
    img_size = (336,336)

# dataloader parameters
batch_size = 2  # actual batch size is this value multiplied by n_transforms(_test)

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 1  # total epochs = meta_epochs * sub_epochs
sub_epochs = 10  # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True
# figure paramaters
nbins1 = 10
nbins2 = 10