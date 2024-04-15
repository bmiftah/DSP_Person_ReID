
# Introduction

 
  **DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION**
  
 We propose a modified model based on learning self-attetnion module disected into  spatial and channel attention which shows a better improvement in the generic classfication tasks. Our Current experiments is carried out on the Person-ReID tasks which is framed as a classification problem. The code here is mainly based on the work in Parameter-Free Spatial Attention Network for Person Re-Identification and PCB.
This code is implementain of the paper entitled DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION. 

## Proposed Model

![Mode design](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Model%20Design.png)

As can be seen from the above model , the re-id task here is framed as classifiaction. Feature is extracted using the main backbone model( shown with light blue color above). Our deep self-attention is embedded before the intermediate supervision shown again with light blue and marked as DSA-1, DSA-2 and DSA-3 . The intermediate attention learning is shown in browan color. As in the earlier work by  The brown boxes shows 6 part classifiers (P). It only appears in the ablation study. Then the total loss is the summation over all deep supervision losses, six part losses and the loss from the backbone. Notice that our DSP is applied to each of the first three blocks of the main model.


## Pre-requiste

Python 3.6, Pytorch 2.0 
## Training the model 
use the below command to pass comman line argument to the main.py , inlcuding dataset path , and other parameters 

!python main.py -d market -b 48 -j 4 --epochs 50 --log logs/market/ --combine-trainval --step-size 40 --data-dir Market-1501  

Running the code might display extra information other than epoch no, and lose. I put those print function to monitor some code but you may comment them all. I run it on colab directly but you can try it on you local machine if you have GPU. 

## Extracting feature map
To extract feature map. Use  single_images_features_3.py and single_features_cam.py  . Make sure to specify path to the image and the trained model. Note that image from which you want to extract feature need to be converted to .npy array before passed on to the extractor. I already have converted some of the sample image and you see some result in the folder named 'sample_features'
## Dataset 
![Market-1501](https://pan.baidu.com/s/1qlCJEdEY7UueGL-VdhH6xw) use password 1ir5 ( I give full credit to the  authors of the paper entitled "Parameter-Free Spatial Attention Network for Person Re-Identification" for availing the data and I am using the same location as mention in thier repo ![here] (https://github.com/XiongDei/Spatial-Attention)
## Ablation Study - sample feature 
Hereunder we show feature extracted by the original Resnet-50 model and the our DSP model . For each pair of images. The one on the left is the original and the one on the right is our model's feature. (Refer that paper for detail discussion

-----------------  Block -1             ------------|----------------------   Block -2  -----------
  


  
![ sample feature map from our DSP model ](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Abalation%20study.png)  

## Ablation Study - pose variation captured 
Our model show resieliance for pose varation as can be seen below 
![pose variation ](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Abalation_study_2.png)


## Ablation Study - correlation among lerned features 
To left side is correlation of feature learned by our model and the one on the left is from the baseline model. As can be seen, feature learned by our model tend to be much correlated as evidenced from the dense region around the diagonal 
![ feature correlation ](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Feature%20correlation%20amount%20features.png)

## Citiaion
If you find this code or part of this helpful for you reserach , please cite our paper and paper Xiong Dei and his coauthors for his kind contribution , More in : https://github.com/XiongDei/Spatial-Attention
