
# Introduction

 
  **DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION**
  
 We propose a modified model based on learning self-attetnion module disected into  spatial and channel attention which shows a better improvement in the generic classfication tasks. Our Current experiments is carried out on the Person-ReID tasks which is framed as a classification problem. The code here is mainly based on the work in Parameter-Free Spatial Attention Network for Person Re-Identification and PCB.
This code is implementain of the paper entitled DEEPLY SUPERVISED SELF-ATTENTION LEARNING MODEL FOR PERSON RE-IDENTIFICATION. 

## Proposed Model

![Mode design](https://github.com/bmiftah/DSP_Person_ReID/blob/main/Model%20Design.png)

As can be seen from the above model , the re-id task here is framed as classifiaction. Feature is extracted using the main backbone model( shown with light blue color above). Our deep self-attention is embedded before the intermediate supervision shown again with light blue and marked as DSA-1, DSA-2 and DSA-3 . The intermediate attention learning is shown in browan color. As in the earlier work by  The brown boxes shows 6 part classifiers (P). It only appears in the ablation study. Then the total loss is the summation over all deep supervision losses, six part losses and the loss from the backbone. Notice that our DSP is applied to each of the first three blocks of the main model.



