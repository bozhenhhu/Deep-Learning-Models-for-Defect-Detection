# Deep-Learning-Models-for-Defect-Detection  
Some methods about automated thermography defects detection  
## 1.Papers  
If you want to know more about the infrared thermal data and some relevant deep learning methods, you could choose to read relevant papers, listed as follows:  
[1] Bozhen Hu, Bin Gao, Wai Lok Woo,etc. “A Lightweight Spatial and Temporal Multi-Feature Fusion Network for Defect Detection,” IEEE Trans. Image Processing, 2020.   
[2]  Lingfeng Ruan,  Bin Gao,  Shichun Wu,  and  Wai Lok Woo,  Joint Loss Structured Deep Adversarial Network for Thermography Defect Detecting System, Neurocomputing, vol. 417, no. 2020, pp. 441-457, Sep 2020  
[3] Qin Luo,  Bin Gao,  Wailok Woo,  and  Yang Yang,  Temporal and spatial deep learning network for infrared thermal defect detection, NDT and E International, vol. 108, no. 102164, pp. 1-13, Aug 2019  
You can find the above papers, and other more papers through this link:  http://faculty.uestc.edu.cn/gaobin/zh_cn/lwcg/153392/list/index.htm  
## 2.Data  
This is flat type data from Optical Pulsed Thermography(OPT) system, you can download it through Google cloud storage by this link:
https://drive.google.com/file/d/1r_x-cFsKaQtXRl5yehhfvLQ3pTXE8DKI/view?usp=sharing  
After download the `data-plane_0.tar.gz`, put this in the `research` folder, then:   
>>tar zxvf plane_0.tar.gz  
>>mkdir pca_imgs   
>>python pca.py  

you can get the pca-processed results in the `pca_imgs` folder for all the data in the `plane_0` folder.  
one of the examples is `N1_1.png`:  
![](https://github.com/bozhenhhu/Deep-Learning-Models-for-Defect-Detection/blob/main/N1_1.png)

When you check the values of these mat files in the `plane_0` folder, if you find some of them are under zero, you may get confused of these negative values, because all of values represent the real-time temperature of one specimen, why the temperature is less than zero? Because every frame of these infrared thermal data was subtracted by it's first frame in order to reduce the influence of the background and noise.
## 3.Dependencies
python version:3.7.9  
tensorflow version:2.2.0  
There are some other dependencies, like matplotlib,opencv-python,etc, you can find these in .py files, when necessary, you can install them. It's pretty easy.

