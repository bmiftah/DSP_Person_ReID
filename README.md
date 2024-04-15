
# Introduction

 
  **DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION**
  
 We propose a modified model based on learning self-attetnion module disected into  spatial and channel attention which shows a better improvement in the generic classfication tasks. Our Current experiments is carried out on the Person-ReID tasks which is framed as a classification problem. The code here is mainly based on the work in Parameter-Free Spatial Attention Network for Person Re-Identification and PCB.
This code is implementain of the paper entitled DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION. 

## Proposed Model

![Mode design](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Model%20Design.png)

As can be seen from the above model , the re-id task here is framed as classifiaction. Feature is extracted using the main backbone model( shown with light blue color above). Our deep self-attention is embedded before the intermediate supervision shown again with light blue and marked as DSA-1, DSA-2 and DSA-3 . The intermediate attention learning is shown in browan color. As in the earlier work by  The brown boxes shows 6 part classifiers (P). It only appears in the ablation study. Then the total loss is the summation over all deep supervision losses, six part losses and the loss from the backbone. Notice that our DSP is applied to each of the first three blocks of the main model.


## Pre-requiste

Python 3.6, Pytorch 2.0 
## Dataset 
![Market-1501](https://pan.baidu.com/s/1qlCJEdEY7UueGL-VdhH6xw) use password 1ir5 ( I give full credit the  authors of the paper entitled "Parameter-Free Spatial Attention Network for Person Re-Identification" for availing the data and I am using the same location as mention in thier repo ![here] (https://github.com/XiongDei/Spatial-Attention)
## Ablation Study - sample feature 
Hereunder we show feature extracted by the original Resnet-50 model and the our DSP model . For each pair of images. The one on the left is the original and the one on the right is our model's feature

![ sample feature map from our DSP model ](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Abalation%20study.png)

## Ablation Study - pose variation captured 
Our model show resieliance for pose varation as can be seen below 
![pose variation ] (https://github.com/bmiftah/DSP_Person_ReID/blob/main/Abalation_study_2.png)


## Ablation Study - correlation among lerned features 
![ feature correlation ] ()
