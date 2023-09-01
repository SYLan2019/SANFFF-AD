import numpy as np
import torch.hub

import config as c
import os
import timm
from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits
import pandas as pd
# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns


def plot(x, colors,class_name = c.class_name):
    save_dir = "./viz/flow/tsne/"
    #save_dir = "./viz/panda/tsne/"
    #save_dir = "./viz/MSAD/tsne/"
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    # 创建调色板，这里返回的是10组颜色，风格是pastel的
    palette = np.array(sns.color_palette("pastel", 2))
    # pastel, husl, and so on

    # Create a scatter plot.
    # 创建800x800大小像素的画布
    f = plt.figure(figsize=(8, 8))
    # 加轴，equal代表等横纵轴比
    ax = plt.subplot(aspect='equal')
    # 这里TSNE降解成2D的了。
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=5, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        # 拿出targets = 0 的 序列假设（n_0,2）沿着轴0求中位数，（返回这两列组成的新数组，组成新的中位数，目标是将target写在一堆数字中间）
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        # 写标签
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        # 设置标签的路径效果，
        txt.set_path_effects([pe.Stroke(linewidth=10, foreground="w"), pe.Normal()])
        # ？
        txts.append(txt)
    save_dir = os.path.join(save_dir, c.dataset)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + '/'+ class_name + '_tsne-pastel.png', dpi=120)
    return f, ax, txts


def plot2(data, x='x', y='y',class_name=c.class_name):
    save_dir = "./viz/flow/tsne/"
    #save_dir = "./viz/panda/tsne/"
    #save_dir = "./viz/MSAD/tsne/"
    # context参数可接收paper、notebook、talk、poster类型 paper < notebook < talk < poster 尺寸
    sns.set_context("notebook", font_scale=1.1)
    # seaborn库中有darkgird（灰色背景+白网格）、whitegrid（白色背景+黑网格）、dark（仅灰色背景）、white（仅白色背景）、ticks（坐标轴带刻度）5种预设的主题。
    sns.set_style("ticks")
    # x,y these should be column names in data.
    #
    sns.lmplot(x=x,
               y=y,
               data=data,
               # 是否根据数据添加回归曲线
               fit_reg=False,
               legend=True,
               height=9,
               # 配合legeng 加入图例
               hue='Label',
               # 传递给plt.scatter的参数，s表示的是大小，alpha表示的是透明度
               scatter_kws={"s": 10, "alpha": 1})

    plt.title('t-SNE Results: Digits', weight='bold').set_fontsize('14')
    plt.xlabel(x, weight='bold').set_fontsize('10')
    plt.ylabel(y, weight='bold').set_fontsize('10')
    save_dir = os.path.join(save_dir,c.dataset)
    os.makedirs(save_dir,exist_ok=True)
    plt.savefig(save_dir+'/'+ class_name +'_tsne-plot2.png', dpi=120)
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

root = "./data/features/"
os.environ['TORCH_HOME'] = 'models\\EfficientNet'
# class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
if __name__ == '__main__':
    class_name = ['frog']
    for n in class_name:
        c.class_name = n
        root = root + c.extractor + '/' + 'z_features' + '/' + 'test' + '/' + c.class_name + '/'
        data = np.load(root + 'testfeatures' + '.npy')
        labels = np.load(os.path.join(root, 'labels.npy'))
        # Place the arrays of data of each digit on top of each other and store in X
        X = np.vstack([data[labels == i] for i in range(2)])
        # Place the arrays of data of each target digit by the side of each other continuosly and store in Y
        Y = np.hstack([labels[labels == i] for i in range(2)])
        Y_1 = list()
        for i in range(10000):
            if i < 1000:
                Y_1.append('normal')
            else:
                Y_1.append('abnormal')
        digits_final = TSNE(perplexity=100).fit_transform(X)
        data = {'x': digits_final[:, 0],
            'y': digits_final[:, 1],
            'Label': Y_1}
        data = pd.DataFrame(data)
        plot2(data,class_name=c.class_name)
        plot(digits_final, Y,class_name=c.class_name)
        print(1)

