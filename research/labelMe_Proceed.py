# -*- coding: utf-8 -*-
"""
#this is used to convert json files labeled by labelme to png/jpg files for some models.

@author: Ryan and bozhen
"""

import json
import os.path as osp
import PIL.Image
import numpy as np
import glob
from labelme import utils


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
    
    
