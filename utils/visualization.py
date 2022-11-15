from easydl import TrainingModeManager, Accumulator
from easydl import variable_to_numpy
import torch.nn.functional as F
from tqdm import tqdm
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib as mpl



def draw_tsne(feature_extractor, classifier, cluster_head,
            s_dl, t_dl, 
            s_label_set, t_label_set,
            writer, dataset='office31'):
    # get features
    prototypes = cluster_head.module.fc.weight.data
    prototypes = prototypes.cpu().numpy()
    N_cluster_head = prototypes.shape[0]
    s_feat,t_feat,s_gt,t_gt = get_embedding(feature_extractor, classifier, s_dl, t_dl)
    
    # tsne
    feat = np.concatenate([prototypes, s_feat, t_feat])
    Y = TSNE(n_components=2).fit_transform(feat)
    y_min, y_max = np.min(Y, 0), np.max(Y, 0)
    Y = (Y - y_min) / (y_max - y_min)
    Y_t = TSNE(n_components=2).fit_transform(t_feat)
    y_min, y_max = np.min(Y_t, 0), np.max(Y_t, 0)
    Y_t = (Y_t - y_min) / (y_max - y_min)

    prototype_Y = Y[:N_cluster_head, :]
    Y = Y[N_cluster_head:, :]

    # tensorboard to visualize
    mode = "text"
    if dataset == 'office31':
        tensorboar_office(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode)
    elif dataset == 'officehome':
        tensorboar_officehome(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode)
    elif dataset == 'visda':
        tensorboard_visda(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode)
    else:
        raise ValueError('wrong dataset input')


def get_embedding(feature_extractor, classifier, s_dl, t_dl):
    with TrainingModeManager([feature_extractor], train=False) as mgr, \
        Accumulator(['s_feat', 's_gt']) as eval_accumulator, \
        torch.no_grad():
        for i, (im, label) in enumerate(tqdm(s_dl, desc='get source feature')):
            s_gt = label.cuda()
            im = im.cuda()
            feature_ex_s = feature_extractor.forward(im)
            before_lincls_feat_s, _ = classifier(feature_ex_s)
            s_feat = F.normalize(before_lincls_feat_s)
            
            val = dict()
            for name in eval_accumulator.names:
                val[name] = variable_to_numpy(locals()[name])    # variable.cpu().data.numpy()
            eval_accumulator.updateData(val)  # eval_accumulator[variable].append()
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]  # variable = eval_accumulator[variable]
    s_feat = val['s_feat']
    s_gt = val['s_gt']

    with TrainingModeManager([feature_extractor], train=False) as mgr, \
            Accumulator(['t_feat', 't_gt']) as eval_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(t_dl, desc='get target feature')):
            t_gt = label.cuda()
            im = im.cuda()
            feature_ex_t = feature_extractor.forward(im)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
            t_feat = F.normalize(before_lincls_feat_t)
            
            val = dict()
            for name in eval_accumulator.names:
                val[name] = variable_to_numpy(locals()[name])    # variable.cpu().data.numpy()
            eval_accumulator.updateData(val)  # eval_accumulator[variable].append()
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]  # variable = eval_accumulator[variable]    
    t_feat = val['t_feat']
    t_gt = val['t_gt']
    return s_feat, t_feat, s_gt, t_gt


def tensorboar_office(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode):
    s_len = len(s_gt)
    t_len = len(t_gt)
    s_feat_Y = Y[:s_len]
    t_feat_Y = Y[s_len:]
    fig = pyplot.figure(1, figsize=(15, 9))
    if mode == "scatter":
        pyplot.switch_backend('agg')
        cmap = mpl.cm.get_cmap("plasma", 20)
        scatter1 = pyplot.scatter(t_feat_Y[:,0], t_feat_Y[:,1], 20, t_gt, '^', cmap=cmap)
        cmap = mpl.cm.get_cmap("Set1", 21)
        scatter2 = pyplot.scatter(s_feat_Y[:,0], s_feat_Y[:,1], 70, s_gt, 'o', cmap=cmap,alpha=0.3)
        pyplot.legend(*scatter2.legend_elements(num=s_label_set))
    if mode == "text":
        char_list = ['0','1','2','3','4','5','6','7','8','9', \
                        'A','B','C','D','E','F','G','H','I','J',\
                            'a','b','c','d','e','f','g','h','i','j','k']
        
        label2index_map_s = {s_label_set[i]:i for i in range(len(s_label_set))}
        label2index_map_t = {t_label_set[i]:i for i in range(len(t_label_set))}
        cmap = mpl.cm.get_cmap("Set1", 20)
        for i in range(s_feat_Y.shape[0]):
            pyplot.text(s_feat_Y[i, 0], s_feat_Y[i, 1], char_list[s_gt[i]], color=cmap(label2index_map_s[s_gt[i]]),
                    fontdict={'weight': 'bold', 'size': 9})
        cmap = mpl.cm.get_cmap("plasma", 21)
        for i in range(t_feat_Y.shape[0]):
            pyplot.text(t_feat_Y[i, 0], t_feat_Y[i, 1], char_list[t_gt[i]], color=cmap(label2index_map_t[t_gt[i]]),
                    fontdict={'weight': 'bold', 'size': 9})
        scatter_star = pyplot.scatter(prototype_Y[:,0], prototype_Y[:,1], s=140, marker='*', c='#cc00ff',alpha=0.3)
        pyplot.xticks([])
        pyplot.yticks([])    
    writer.add_figure(tag=f'tsne_visualization', figure=fig)
    writer.close()

def tensorboar_officehome(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode):
    s_len = len(s_gt)
    t_len = len(t_gt)
    s_feat_Y = Y[:s_len]
    t_feat_Y = Y[s_len:]
    
    officehome_main_color(1, writer, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y)
    officehome_partial_color(2, writer, 15, 34, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y)
    officehome_partial_color(3, writer, 35, 54, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y)
    officehome_partial_color(4, writer, 55, 64, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y)
    
    writer.close()

def officehome_main_color(fig_id, writer, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y):
    fig = pyplot.figure(fig_id, figsize=(15, 9))
    s_char_list = ['0','1','2','3','4','5','6','7','8','9', \
                    'A','B','C','D','E']
    label2index_map_s = {s_label_set[i]: i for i in range(len(s_label_set))}
    
    cmap = mpl.cm.get_cmap("Set1", 15)
    for i in range(s_feat_Y.shape[0]):
        pyplot.text(s_feat_Y[i, 0], s_feat_Y[i, 1], s_char_list[s_gt[i]], color=cmap(label2index_map_s[s_gt[i]]),
                fontdict={'weight': 'bold', 'size': 9})
    cmap = mpl.cm.get_cmap("plasma", 10)
    # select out common
    common_ids = np.argwhere(t_gt < 10).reshape(-1)
    tp_ids = np.argwhere(t_gt > 9).reshape(-1)
    common_t_feat_Y = t_feat_Y[common_ids]
    tp_t_feat_Y = t_feat_Y[tp_ids]
    common_t_gt = t_gt[common_ids]
    tp_t_gt = t_gt[tp_ids]
    t_common_char_list = ['0','1','2','3','4','5','6','7','8','9']
    label2index_map_t_common = {t_label_set[i]: i for i in range(10)}
    for i in range(common_t_feat_Y.shape[0]):
        pyplot.text(common_t_feat_Y[i, 0], common_t_feat_Y[i, 1], t_common_char_list[common_t_gt[i]], 
                color=cmap(label2index_map_t_common[common_t_gt[i]]),
                fontdict={'weight': 'bold', 'size': 9})
    
    scatter = pyplot.scatter(tp_t_feat_Y[:,0], tp_t_feat_Y[:,1], s=70, marker='^', c='#a1a3a6', alpha=0.3)
    scatter_star = pyplot.scatter(prototype_Y[:,0], prototype_Y[:,1], s=140, marker='*', c='#cc00ff', alpha=0.3)
    pyplot.xticks([])
    pyplot.yticks([])    
    writer.add_figure(tag=f'15-64 grey', figure=fig)


def officehome_partial_color(fig_id, writer, start, end, s_label_set, s_feat_Y, s_gt, t_feat_Y, t_gt, t_label_set, prototype_Y):
    # 15-34 colorful
    all_list = [i for i in range(15, 65)]
    colorful_list = [i for i in range(start, end+1)]
    uncolorful_list = sorted(set(all_list) - set(colorful_list))
    fig = pyplot.figure(fig_id,figsize=(15, 9))
    s_char_list = ['0','1','2','3','4','5','6','7','8','9', \
                    'A','B','C','D','E']
    
    label2index_map_s = {s_label_set[i]:i for i in range(len(s_label_set))}
    
    cmap = mpl.cm.get_cmap("Set1", 15)
    for i in range(s_feat_Y.shape[0]):
        pyplot.text(s_feat_Y[i, 0], s_feat_Y[i, 1], s_char_list[s_gt[i]], color=cmap(label2index_map_s[s_gt[i]]),
                fontdict={'weight': 'bold', 'size': 9})
    cmap = mpl.cm.get_cmap("plasma", 10)
    # select out common
    common_ids = np.argwhere(t_gt < 10).reshape(-1)
    
    common_t_feat_Y = t_feat_Y[common_ids]
    common_t_gt = t_gt[common_ids]
    
    t_common_char_list = ['0','1','2','3','4','5','6','7','8','9']
    label2index_map_t_common = {t_label_set[i]:i for i in range(10)}
    for i in range(common_t_feat_Y.shape[0]):
        pyplot.text(common_t_feat_Y[i, 0], common_t_feat_Y[i, 1], t_common_char_list[common_t_gt[i]], 
                color=cmap(label2index_map_t_common[common_t_gt[i]]),
                fontdict={'weight': 'bold', 'size': 9})
    
    color_tp_ids_1 = np.argwhere(t_gt > start-1)
    color_tp_ids_2 = np.argwhere(t_gt < end+1)
    cat_color_tp_ids = np.concatenate([color_tp_ids_1, color_tp_ids_2], 0)
    unq,cnt = np.unique(cat_color_tp_ids, return_counts=True)
    indices = ((cnt-1) > 0).reshape(-1)
    color_tp_ids = unq[indices]

    color_tp_t_feat_Y = t_feat_Y[color_tp_ids]
    color_tp_t_gt = t_gt[color_tp_ids]
    label2index_map_color_tp = {colorful_list[i]: i for i in range(end-start+1)}
    cmap = mpl.cm.get_cmap("coolwarm", end-start)    # change colormap
    scatter1 = pyplot.scatter(color_tp_t_feat_Y[:,0], color_tp_t_feat_Y[:,1], 70, color_tp_t_gt, '^', 
                            cmap=cmap, alpha=0.5)
    # uncolorful_list
    uncolor_tp_ids = None
    for id in uncolorful_list:
        t = np.argwhere(t_gt == id).reshape(-1, 1)
        if uncolor_tp_ids is None:
            uncolor_tp_ids = t
        uncolor_tp_ids = np.concatenate([uncolor_tp_ids, t], 0)
    uncolor_tp_ids = uncolor_tp_ids.reshape(-1)
    uncolor_tp_t_feat_Y = t_feat_Y[uncolor_tp_ids]
    uncolor_tp_t_gt = t_gt[uncolor_tp_ids]
    scatter2 = pyplot.scatter(uncolor_tp_t_feat_Y[:,0], uncolor_tp_t_feat_Y[:,1], s=70, \
                            marker='^', c='#a1a3a6', alpha=0.3)
    scatter_star = pyplot.scatter(prototype_Y[:,0], prototype_Y[:,1], s=140, marker='*', c='#cc00ff',alpha=0.3)
    pyplot.xticks([])
    pyplot.yticks([])    
    writer.add_figure(tag=f'{start}-{end} colorful', figure=fig)


def tensorboard_visda(s_label_set, t_label_set, s_gt, t_gt, Y, prototype_Y, writer, mode):
    s_len = len(s_gt)
    t_len = len(t_gt)
    s_feat_Y = Y[:s_len]
    t_feat_Y = Y[s_len:]
    fig = pyplot.figure(1, figsize=(15, 9))
    if mode == "scatter":
        pyplot.switch_backend('agg')
        cmap = mpl.cm.get_cmap("plasma", 9)
        scatter1 = pyplot.scatter(t_feat_Y[:,0], t_feat_Y[:,1], 20, t_gt, '^', cmap=cmap)
        cmap = mpl.cm.get_cmap("Set1", 9)
        scatter2 = pyplot.scatter(s_feat_Y[:,0], s_feat_Y[:,1], 70, s_gt, 'o', cmap=cmap,alpha=0.3)
        pyplot.legend(*scatter2.legend_elements(num=s_label_set))
    if mode == "text":
        char_list = ['0','1','2','3','4','5', \
                        'A','B','C',\
                            'a','b','c']
        label2index_map_s = {s_label_set[i]:i for i in range(len(s_label_set))}
        label2index_map_t = {t_label_set[i]:i for i in range(len(t_label_set))}
        cmap = mpl.cm.get_cmap("Set1", 9)
        for i in range(s_feat_Y.shape[0]):
            pyplot.text(s_feat_Y[i, 0], s_feat_Y[i, 1], char_list[s_gt[i]], color=cmap(label2index_map_s[s_gt[i]]),
                    fontdict={'weight': 'bold', 'size': 9})
        cmap = mpl.cm.get_cmap("plasma", 9)
        for i in range(t_feat_Y.shape[0]):
            pyplot.text(t_feat_Y[i, 0], t_feat_Y[i, 1], char_list[t_gt[i]], color=cmap(label2index_map_t[t_gt[i]]),
                    fontdict={'weight': 'bold', 'size': 9})
        scatter_star = pyplot.scatter(prototype_Y[:,0], prototype_Y[:,1], s=140, marker='*', c='#cc00ff',alpha=0.3)
        pyplot.xticks([])
        pyplot.yticks([])    
    writer.add_figure(tag=f'tsne_visualization', figure=fig)
    writer.close() 