# import torch
# import torch.nn as nn
# from torchvision import models
#
# original_model_alex = models.alexnet(pretrained=False)
# original_model_resnet = models.resnet50(pretrained=False)
#
#
# #print(" Original Model : ", original_model)
#
# class AlexNetConv4(nn.Module):
#             def __init__(self):
#                 super(AlexNetConv4, self).__init__()
#                 self.features = nn.Sequential(
#                     # stop at conv4
#                     *list(original_model_alex.features.children())[:-3]
#                 )
#             def forward(self, x):
#                 x = self.features(x)
#                 return x
#
# model_alex = AlexNetConv4()
#
# #print(" Modified model ",model)
#
# class ResNetConv4(nn.Module):
#     def __init__(self):
#         super(ResNetConv4, self).__init__()
#         self.features = nn.Sequential(
#             # stop at conv4
#             #*list(original_model_resnet.features.children())[:-3]
#
#             *list(original_model_resnet.children())[:-3]
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         return x
#
# model_resnet = ResNetConv4()
#
# print( " ---- ")

# matplotlib inline
import torch
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint, save_checkpoint
import matplotlib as plt

from reid.utils import to_torch

# from reid import models
# from torch.utils.data import DataLoader

#  image and mask input

# image_path = args.image_path
# mask_path = args.mask_path
# mask = cv2.imread(args.mask_path,1)
# img = cv2.imread(args.image_path,1)

image = Image.open(
    "/content/drive/MyDrive/Spatial-Attention-master/Market-1501/query/0001_c1s1_001051_00.jpg")  # Image
# mask = Image.open(
#     "/home/miftah/PycharmProjects/Spatial-Attention-master_2/Market-1501_mask/query/0001_c1s1_001051_00.png")  # mask
# # /content/drive/MyDrive/Spatial-Attention-master
# # imshow(image)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor,
# and normalize to -1..1 with mean/std for ImageNet

preprocess = transforms.Compose([
    # transforms.Resize((384,128)),
    T.RectScale(384, 128),
    T.ToTensor(),
    normalize
])

display_transform = transforms.Compose([
    transforms.Resize((384, 128))])

# mask_PIL = transforms.ToPILImage(mask)
To_tensor = transforms.ToTensor()
# mask_nd_array = np.asarray(mask)
# mask_tensor = To_tensor(mask_nd_array)

# # tensor_mask = mask_tensor.repeat(3, 1, 1)
to_PIL = transforms.ToPILImage()
image2 = np.asarray(image)


# mask_pil = mask_PIL(tensor_mask)
#
tensor = preprocess(image)  # [3,384,128]
#
# tensor_mask = preprocess(mask_pil)

# tensor_mask = tensor_mask.repeat(3,1,1)


prediction_var = Variable((tensor.unsqueeze(0)), requires_grad=True)  # [1,3,384,128]

input = prediction_var  # [1,3,384,128]

model = models.resnet18(pretrained=True)

print(" SA Model  : - ", model)

model.eval()


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


# final_layer = model._modules.get('layer4')



def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret


# local_conv_layer2 = model._modules.get('local_conv_layer2')

target_layer = model._modules.get('layer4')  # layer4 [0,1]


activated_features = SaveFeatures(target_layer)  # [features : (1,512,12,4) hook :(id - > 0).(next id -> 1) ]

# The below line cause - IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
# x_attn1 = self.SA1(x_layer1,xm,flag_1
# h = x.size(2)  # 192 - conv1

prediction = model(prediction_var)  # prediction  [1,1000]


pred_probabilities = F.softmax(prediction).cpu().data.squeeze()  # [1000]
activated_features.remove()

print(topk(pred_probabilities, 1))


# cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
# ValueError: shapes (256,) and (2048,192) not aligned: 256 (dim 0) != 2048 (dim 0)
def heatmap2d(img, ht_map):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Image")
    ax1 = fig.add_subplot(122, title="Heatmap")
    ax2 = fig.add_subplot(122, title="Heatmap_fused")

    # ax0.imshow(img)
    # ax1.imshow(ht_map, cmap='jet')
    # heatmap=ax1.imshow(ht_map)
    # fig.colorbar(heatmap)

    # fig.savefig('./heatmap/heatmap'+str(i))
    fig.savefig('./heatmap/heatmap_12_12')
    ht_map_resized = np.resize(ht_map, (np.shape(img)))
    # img=Image.fromarray(img)
    ht_map_resized = torch.Tensor(ht_map_resized)
    img = torch.Tensor(img)
    img_heatmap = np.uint8(0.5 * img + 0.5 * ht_map_resized)
    ax1.imshow(img_heatmap)
    ax0.imshow(img)
    # ax1.imshow(heatmap_fused)
    plt.show()



def getCAM(img, feature):
    
    b, nc, h, w = feature.shape

    display_transform2 = transforms.Compose([
         transforms.ToTensor()]) #transforms.Resize((h, w)),

    to_PIL = transforms.ToPILImage()
    img = to_PIL(img)

    img = display_transform2(img)

    print("image : ",img.shape)
    print(" feature ",feature.shape)

    feature = cv2.applyColorMap(np.uint8(255 * feature), cv2.COLORMAP_JET)
    feature = np.float32(feature) / 255

    #img_heatmap = np.uint8(0.5 * img + 0.5 * feature)
    # cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))

    #feature = feature.reshape(h, w)
    feature = feature - np.min(feature)
    feature = (feature / np.max(feature))*255

    #img_heatmap = np.uint8(0.5 * img + 0.5 * ht_map_resized)
    cv2.imwrite("/content/drive/MyDrive/Spatial-Attention-master/heatmap_demo/cam_demo.jpg", feature[0])
    return [feature]


weight_softmax_params = list(model._modules.get('fc').parameters())  # [0 -> (1000,512) , 1 -> (1000)]
# weight_softmax_params = list(model._modules.get('instance6').parameters())
# print("weight_softmax_params:",weight_softmax_params)
# print("weight_softmax_params[0]",weight_softmax_params[0])
# weight_softmax_params = list(model._modules.get('instance6').parameters())

weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())  # [1000,512]

# weight_softmax_params

class_idx = topk(pred_probabilities, 1)[1].int()

overlay = getCAM(image2,activated_features.features)

imshow(overlay[0], alpha=0.5, cmap='jet')  # [12,4]

imshow(display_transform(image))

imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.75, cmap='jet')

# print(classmethod)
print(" .... ")















































