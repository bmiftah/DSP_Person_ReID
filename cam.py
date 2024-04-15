print(" Class Activation Map Demo")
#matplotlib inline
import torch
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils import to_torch
#from reid import models
#from torch.utils.data import DataLoader

#  image and mask input

# image_path = args.image_path
# mask_path = args.mask_path
# mask = cv2.imread(args.mask_path,1)
# img = cv2.imread(args.image_path,1)

image = Image.open("/content/drive/MyDrive/Spatial-Attention-master/Market-1501/query/0001_c1s1_001051_00.jpg")  # Image
# mask = Image.open("/home/miftah/PycharmProjects/Spatial-Attention-master_2/Market-1501_mask/query/0001_c1s1_001051_00.png")            # mask
# /content/drive/MyDrive/Spatial-Attention-master/Market-1501/query/0001_c1s1_001051_00.jpg


#imshow(image)
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
   transforms.Resize((384,128))])


#mask_PIL = transforms.ToPILImage(mask)
To_tensor = transforms.ToTensor()
#mask_nd_array = np.asarray(mask)
#mask_tensor = To_tensor(mask_nd_array)


#tensor_mask = mask_tensor.repeat(3,1,1)
mask_PIL = transforms.ToPILImage()
#test_image = Image.open(image).convert('RGB')

# mask_pil = mask_PIL(tensor_mask)
#
tensor = preprocess(image)  #  [3,384,128]
#
# tensor_mask = preprocess(mask_pil)

#tensor_mask = tensor_mask.repeat(3,1,1)


prediction_var = Variable((tensor.unsqueeze(0)), requires_grad=True)  # [1,3,384,128]


input = prediction_var  # [1,3,384,128]

model = models.resnet18(pretrained=True)
# model_path = 'checkpoint.pth.tar'
# # load model
# checkpoint = load_checkpoint(model_path)
# model_dict = model.state_dict()
# checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
# model_dict.update(checkpoint_load)
# model.load_state_dict(model_dict)
#
#
# print(" SA Model  : - ",model)

model.eval()

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

# final_layer = model._modules.get('layer4')

# target_1 = base._modules.get('layer4')
# local_conv_layer2 = model._modules.get('local_conv_layer2')
# local_conv_layer3 = model._modules.get('local_conv_layer3')
# local_conv_layer4 = model._modules.get('local_conv_layer4')
#
# for name, module in base._modules.items():
#     if name == 'layer4':
#         target_layer = 'layer4'
#         target_module = module
#         # print(" name   : ",name)
#         # print(" module : ",module)
#         break
# print(" base layer 4",local_conv_layer4)

# class Trainer(BaseTrainer):
#     def _parse_data(self, inputs):
#         imgs, _, pids, _ = inputs
#         inputs = Variable(imgs)
#         targets = Variable(pids.cuda())
#         return inputs, targets
#
#     def _forward(self, inputs,input_m,targets):
#         # model called here
#         # input shape - [48,3,384,128]
#         #inputs = [inputs,input_m]
#         outputs = self.model(inputs,input_m)
#         index = (targets-751).data.nonzero().squeeze_()

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


#local_conv_layer2 = model._modules.get('local_conv_layer2')

final_layer = model._modules.get('layer4')  #  layer4 [0,1]

#final_layer_2 = base._modules.get('local_conv_layer3')
#local_conv_layer3 = model._modules.get('local_conv_layer3')
#local_conv = model._modules.get('local_conv') #  layer 4


#instance3 = model._modules.get('instance3') # FC layer that takes -> local_conv_layer3
#instance6 = model._modules.get('instance6') # FC layer that takes -> local_conv_layer4 ( x - layer-4 .

#print(instance3)


#print(" main layer4 ",final_layer)
activated_features = SaveFeatures(final_layer) # [features : (1,512,12,4) hook :(id - > 0).(next id -> 1) ]

# The below line cause - IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
# x_attn1 = self.SA1(x_layer1,xm,flag_1
# h = x.size(2)  # 192 - conv1

prediction = model(prediction_var)  # prediction  [1,1000]
# DO something to pic proper argument for F.softmax function ..
#prediction_2 = prediction[0]
#prediction_3 = (prediction[1])[9]

# use instance-x as fc layer here and send the result to the below softmax unit
#instance6 = model._modules.get('instance6')
# print(" Layer - 3 ")
# print(" local_conv_layer3 : ",local_conv_layer3)
# print(" instance3         : ",instance3)
# print(" Layer - 4 ")
# print(" local_conv        : ",local_conv)
# print(" instance6         : ",instance6)

pred_probabilities = F.softmax(prediction).cpu().data.squeeze() # [1000]
activated_features.remove()

print(topk(pred_probabilities,1))

# cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
# ValueError: shapes (256,) and (2048,192) not aligned: 256 (dim 0) != 2048 (dim 0)

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    #cv2.imwrite("heatmapReSAnet/cam_1.jpg", cam_img[0])
    return [cam_img]
def heatmap2d(img, ht_map):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Image")
    ax1 = fig.add_subplot(122, title="Heatmap")
    ax2 = fig.add_subplot(122, title="Heatmap_fused")

    # ax0.imshow(img)
    # ax1.imshow(ht_map, cmap='viridis')
    #heatmap=ax1.imshow(ht_map)
    #fig.colorbar(heatmap)

    # fig.savefig('./heatmap/heatmap'+str(i))
    fig.savefig('./heatmap/heatmap_12_12')
    ht_map_resized = np.resize(ht_map,(np.shape(img)))
    # img=Image.fromarray(img)
    ht_map_resized = torch.Tensor(ht_map_resized)
    img = torch.Tensor(img)
    img_heatmap = np.uint8(0.5*img + 0.5*ht_map_resized)
    ax1.imshow(img_heatmap)
    ax0.imshow(img)
    # ax1.imshow(heatmap_fused)
    plt.show()
    # fig.savefig('./heatmap/heatmap_fused')




weight_softmax_params = list(model._modules.get('fc').parameters())  # [0 -> (1000,512) , 1 -> (1000)]
#weight_softmax_params = list(model._modules.get('instance6').parameters())
#print("weight_softmax_params:",weight_softmax_params)
#print("weight_softmax_params[0]",weight_softmax_params[0])
#weight_softmax_params = list(model._modules.get('instance6').parameters())

weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy()) # [1000,512]

#weight_softmax_params

class_idx = topk(pred_probabilities,1)[1].int()


overlay = getCAM(activated_features.features, weight_softmax, class_idx )
imshow(overlay[0])
# #img = display_transform(skimage.transform.resize(image, tensor.shape[1:3]))
# imshow(image)
#
# imshow(overlay[0], alpha=0.5, cmap='jet') # [12,4]
# imshow(display_transform(image))
# imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.45, cmap='jet')
#
# #heatmap2d(skimage.transform.resize(overlay[0], tensor.shape[1:3]))
# imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.45, cmap='jet')
#
# cam_img = skimage.transform.resize(overlay[0], tensor.shape[1:3])
# imshow(cam_img,alpha=0.45, cmap='jet')
#cv2.imwrite("heatmap_demo/22.jpeg",cam_overlayed) # TypeError: Expected Ptr<cv::UMat> for argument 'img'
#print(classmethod)
print(" .... ")
print(" GPU  ? ",torch.cuda.is_available())
dx, dy = 0.05, 0.05

y = np.arange(-12, 12, dx)
x = np.arange(-4.0, 4.0, dy)
extent = np.min(x), np.max(x), np.min(y), np.max(y)

fig = plt.figure()

ax0 = fig.add_subplot(131, title="Image")
ax1 = fig.add_subplot(132, title="Heatmap")
ax2 = fig.add_subplot(133, title="Heatmap_Overlayed")

ax0.imshow(display_transform(image))
ax1.imshow(overlay[0], alpha=0.5, cmap='jet')
ax2.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.7, cmap='jet')

# ax0.imshow(overlay[0], alpha=0.5, interpolation='gaussian', cmap=plt.cm.jet, extent=extent)
# ax1.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.45, interpolation='gaussian', cmap=plt.cm.jet, extent=extent)
# ax2.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.7, interpolation='gaussian', cmap=plt.cm.jet, extent=extent)
# # dx, dy = 0.05, 0.05
#
# y = np.arange(-8, 8, dx)
# x = np.arange(-4.0, 4.0, dy)
#fig.colorbar(heatmap)

# fig.savefig('./heatmap/heatmap'+str(i))
fig.savefig('/content/drive/MyDrive/Spatial-Attention-master/heatmap_demo/sample.jpeg')















































