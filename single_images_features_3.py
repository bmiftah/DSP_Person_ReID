# Source - https://www.programmersought.com/article/13453376400/

# Source - https://www.programmersought.com/article/13453376400/
import os
import cv2
import numpy as np
import torch
from torch import utils
from torch.autograd import Variable
from torchvision import models ,transforms,datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import PIL
import math
from reid import models
import argparse

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    print("")
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (384, 128))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return [im_as_var , im_as_ten]



class FeatureVisualization():
    def __init__(self, img_path, selected_layer,target_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        #self.pretrained_model = models.vgg16(pretrained=True).features
        #self.pretrained_model2 = models.resnet50(pretrained=True)
        self.target_layer = target

        model_path = './SA_trained_model/checkpoint_original.pth.tar'
        arch = 'resnet50'
        features = 256
        dropout = 0.47
        num_classes = 751
        model = models.create(arch, num_features=features, dropout=dropout, num_classes=num_classes)

        # checkpoint = load_checkpoint(model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)

        self.pretrained_mode3 = model.eval()

    def process_image(self):
        img_raw = cv2.imread(self.img_path)
        img = preprocess_image(img_raw)
        return img


    def get_feature_2(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        #model_resnet = self.pretrained_model2
        model_sa = self.pretrained_mode3
        input2 = self.process_image()

        print(" input2 shape for Resnet50 ",input2[0].shape)
        x1 = input2[0]
        #target_layer = self.pretrained_model2._modules.get('layer1')
        base = model_sa._modules.get('base')

        for name,module in base._modules.items():
            x1 = module(x1)
            print(" module name ",name)
            if name == self.target_layer:
             #target_module = module
             print(" Target name   : ", name)
             #print(" \n Target module : ", module)
             return x1





    def get_single_feature_2(self):

        features2 = self.get_feature_2()
        print(" Resnet50 target layer feature -> ", features2.shape) # torch.Size([1, 256, 56, 56])

        feature0 = features2[:,1,:,:]
        #feature1 = features2[:,1,:,:]
        #feature2 = features2[:,2,:,:]

        #feature_2 = torch.cat([feature0,feature1,feature2],dim=0)
        #print(" Feature 0,1,2 ",feature_2.shape)

        features2 = features2[:,0,:,:]
        #print(features2)
        print(" shape of selected feature map Resnet50 ",features2.shape)
        features2 = features2.view(features2.shape[1],features2.shape[2])
        print(" selected feature map reshap ",features2.shape)

        return features2





    def save_feature_to_img_2(self):
        # to numpy

        feature2 = self.get_single_feature_2()
        feature2 = feature2.data.numpy()

        # Image to tensor and to numpy
        print(" feature2 shape ",feature2.shape)
        w,h = feature2.shape[0],feature2.shape[1]


        img_raw1 = Image.open(self.img_path)


        img_raw_np = np.array(img_raw1)
        print(" Image shape ", img_raw_np.shape)
        #imshow(img_raw_np)

        # img_raw_resize = cv2.resize(img_raw, (h,w))
        #imshow(img_raw_resize)


        input_imge = self.process_image()
        input_imge = input_imge[1]
        newsize = (h,w)
        img_raw_resize = img_raw1.resize(newsize)
        img_raw_np_resize = np.array(img_raw_resize)
        #imshow(img_raw_np_resize)


        print(" feature and inputshape for fusion : ")
        print(" feature shape ", feature2.shape)
        print(" input_image   ", img_raw_np_resize.shape)


        # use sigmod to [0,1]
        #feature2 = 1.0 / (1 + np.exp(-1 * feature2))

        # to [0,255]
        # feature = np.round(feature * 255)
        # print(feature[0])
        #feature22 =
        #feature22 = cv2.applyColorMap(np.uint8(np.round(255 * feature2)), cv2.COLORMAP_JET)

        PIL.Image.fromarray(img_raw_np_resize.astype(np.uint8)).save('png_img_feature/0467.img.png')
        PIL.Image.fromarray((feature2 * 255).astype(np.uint8)).save('png_img_feature/0467.feat.png')




        #display setup
        dx, dy = 0.05, 0.05

        y = np.arange(-8, 8, dx)
        x = np.arange(-4.0, 4.0, dy)
        X, Y = np.meshgrid(x, y)
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        alpha = 0.3
        #feature2 = np.uint8(np.round(255.5 * feature2))
        #alpha = 0.5
        #img_feature = (feature2[:, :, None].astype(np.float64) * 255.0*alpha + img_raw_np_resize * (1 - alpha)).astype(np.float64)

        #feature2 = (255.5 * feature2 / np.amax(feature2)).astype(np.uint8)
        #alpha = 0.5

        # print(" center of feature array ")
        # print(feature2[feature2.shape[0] // 2 - 5: feature2.shape[0] // 2 + 5, feature2.shape[1] // 2 - 5: feature2.shape[1] // 2 + 5])
        # print(" center of image array ")
        # print(img_raw_np_resize[img_raw_np_resize.shape[0] // 2 - 5: img_raw_np_resize.shape[0] // 2 + 5,img_raw_np_resize.shape[1] // 2 - 5: img_raw_np_resize.shape[1] // 2 + 5])


        img_feature = ((plt.cm.jet(feature2)[:, :, :3] * 255.0)*alpha + img_raw_np_resize*(1-alpha)).astype(np.uint8)
        #img_feature = ((plt.cm.jet(feature2)[:, :, :3] * 255) * alpha + img_raw_np_resize * (1 - alpha)).astype(np.uint8)

        #img_heatmap2 = np.uint8(0.5 * img_raw_np_resize + 0.5 * feature2*255.0)

        fig = plt.figure()
        ax0 = fig.add_subplot(131, title="Image")
        ax1 = fig.add_subplot(132, title="Heatmap")
        ax2 = fig.add_subplot(133, title="overlayed")
        #ax3 = fig.add_subplot(144, title="feature22")


        ax0.imshow(img_raw1,alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet,extent=extent)
        ax1.imshow(feature2,alpha = 0.75, interpolation = 'gaussian', cmap = plt.cm.jet,extent =extent)
        ax2.imshow(img_feature, alpha =1.,interpolation = 'gaussian', cmap = plt.cm.jet,extent=extent)
        #ax3.imshow(feature22, alpha=0.6, interpolation='gaussian', cmap=plt.cm.jet, extent=extent)
        plt.show()

        # show images
        #
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(img_raw_np_resize)
        # axs[0, 1].imshow(feature2, alpha=1., interpolation='gaussian', cmap=plt.cm.jet)
        # axs[1, 0].imshow(img_feature, alpha=1., interpolation='gaussian', cmap=plt.cm.jet)
        # axs[1, 1].remove()
        # plt.show()

        cv2.imwrite('./img_renet50.jpg', feature2)


if __name__ == '__main__':
    # get class
    target = 'layer1'
    img_path ='Market-1501/bounding_box_train/0046_c5s1_004026_04.jpg'
    myClass = FeatureVisualization('./input_images/sample_input.jpg', 5,target)
    #myClass = FeatureVisualization(img_path, 5, target)
    #print(myClass.pretrained_model)

    #myClass.save_feature_to_img()
    myClass.save_feature_to_img_2()


