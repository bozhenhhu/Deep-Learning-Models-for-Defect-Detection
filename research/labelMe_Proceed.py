# -*- coding: utf-8 -*-
"""
#this is used to convert json files labeled by labelme to png/jpg files for some models.
first using Labelme to get json files
second modify hyperparameters in this file named label_dir and json_file
third run this procedure to get images
last suggesting binarize the resulted ground truth; is's easy, like 
   labels[labels<0.5] = 0
   labels[labels!=0] = 1

@author: Ryan and bozhen
"""

import json
import os.path as osp
import PIL.Image
import numpy as np
import glob
from labelme import utils

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def labelcolormap(N=256):
    warnings.warn('labelcolormap is deprecated. Please use label_colormap.')
    return label_colormap(N=N)

def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

def draw_label(label, img, label_names, colormap=None):
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if colormap is None:
        colormap = label_colormap(len(label_names))

    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')
    return label_viz



#单步操作，调试方便
# =============================================================================
# out_dir =  r'F:\C2403\testdata\rgbimg'
# json_file = r'F:\C2403\testdata\rgbimg\D1\D1_17.json'
# data = json.load(open(json_file))
#  
# img = utils.img_b64_to_arr(data['imageData'])
# lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
# 
# img2 = 80*np.ones((656,875,3), dtype = np.uint8)
# lbl_viz = draw_label(lbl, img2, lbl_names)
#  
# PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
# PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
# PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
# 
# =============================================================================
    
#black means defect，white means background  


label_dir = r''  #path to save the generated img labels
json_file = r''  #path to json files
count = 1
for jso in (glob.glob(json_file + '/*.json')):
    data = json.load(open(jso))
    img = utils.img_b64_to_arr(data['imageData']) 
    a,b,c = img.shape
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    img2 = 80*np.ones((a,b,c), dtype = np.uint8)
    lbl_viz = draw_label(lbl, img2, lbl_names)
    im_at = np.ones((a,b), dtype = np.uint8)
    for i in range(a):
        for j in range(b):
            if lbl_viz[i,j,0] < 80:
                im_at[i,j] = 255 
            else:
                im_at[i,j] = 0
    PIL.Image.fromarray(im_at).save(label_dir + '/' + str(count) + '.jpg')
    count = count + 1
    
    
