## This is an introduction file to all files in the `research` floder  
## 1.LSTM.py   
This file includes a 3-layer LSTM model and training procedure,however, it's not intact, you need to re-write the `get_data` function to get train and label data, feeded in the network later.  
## 2.attention.py  
It contains some attention blocks, including SE, sc_SE, st_Attention, and ECA  
## 3.augmentation.py  
some augmentation functions  
## 4.convert_mat_data.m  
After obtaining csv data, you may need to convert it into .mat format data, after that, if necessary, using this procedure to convert .mat into .mat to make python can read .mat files properly.  
## 5.focal_loss.py  
It is the focal loss, it uses Keras and tensorflow<2.0  
## 6.hparams.py
It contains some hyperparameters for models
## 7.Kmeans.py  
One of cluster methods, operate on a single image
## 8.labelMe_proceed.py
If you use `labelme` to label data, you can get json files, if you want to convert json files into png/jpg format, you can use this file.
## 9.mat_process.py
This is the main procedure to process training files , train and test UNet++ model.   
The trained model named huv27.h5, which can be found through Baidu cloud storage:  
链接：https://pan.baidu.com/s/1_RfwP52neXcWd2DHgqW8xw   
提取码：51s6   
复制这段内容后打开百度网盘手机App，操作更方便哦  
If you set `args.mode` not equal to 0, it runs the test prsocedure, one of the results is hu27_xy_20200812_007g_25_1.bmp    
![](https://github.com/bozhenhhu/Deep-Learning-Models-for-Defect-Detection/blob/main/research/hu27_xy_20200812_0007g_25_1.bmp)  
## 10.model_test.m  
It uses Matlab to load trained models(eg:huv27.h5), and rewrite data process codes depended on Python codes in `mat_process.py`      
huv25.h5 and one test data defectnetinput.csv can be got by the above link.    
The output images of this procedure are showing as:    
![](https://github.com/bozhenhhu/Deep-Learning-Models-for-Defect-Detection/blob/main/research/matlab_test.png)    
![](https://github.com/bozhenhhu/Deep-Learning-Models-for-Defect-Detection/blob/main/research/matlab_test_result.png)  
## 11.models.py
It has a few deep learning models, like UNet, UNet++ 
## 12.pca.py
It can be run independently to apply `PCA` .
## 13.region_grow.py 
It is a region-growing method
## 14.utils.py
It contains several functions having special effects. Like plotting temperature changing curve, change png to jpg, draw rectangles on am image...
