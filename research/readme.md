## This is an introduction file to all files in the `research` floder  
## 1.LSTM.py   
This file include 3-layer LSTM model and train procedure,however, it's not intact, you need to re-write `get_data` function to get train and label data in order to feed in the network.  
## 2.attention.py  
It contains some attention blocks, including SE, sc_SE, st_Attention, and ECA  
## 3.augmentation.py  
some augmentation functions  
## 4.convert_mat_data.m  
After obtaining csv data, you may need to convert it into .mat format data, after that, if necessary, using this procedure to convert .mat into .mat to make python can read .mat files properly.  
## 5.focal_loss.py  
It is focal loss, it uses Keras and tensorflow<2.0  
## 6.hparams.py
It contains some hyperparameters for models
## 7.Kmeans.py  
one of cluster methods, operate on a single image
## 8.labelMe_proceed.py
If you use `labelme` to label data, you can get json files, if you want to convert json files into png/jpg format, you can use this file.
## 9.mat_process.py
This is the main procedure to process training files , train and test UNet++ model.   
If you set `args.mode` not equal to 0, it run test procedure, one of the results is hu27_xy_20200812_007g_25_1.bmp  
![](https://github.com/bozhenhhu/Deep-Learning-Models-for-Defect-Detection/blob/main/research/hu27_xy_20200812_0007g_25_1.bmp)
