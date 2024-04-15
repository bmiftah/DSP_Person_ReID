from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch import nn
from torch import autograd


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# class SpatialAttn(nn.Module):
#     """Spatial Attention Layer"""
#     def __init__(self):
#         super(SpatialAttn, self).__init__()
#
#     def forward(self, x):
#         # global cross-channel averaging # e.g. 32,2048,24,8
#         x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
#         h = x.size(2)
#         w = x.size(3)
#         x = x.view(x.size(0),-1)     # e.g. 32,192
#         z = x
#         for b in range(x.size(0)):
#             z[b] /= torch.sum(z[b])
#         z = z.view(x.size(0),1,h,w)
#         return z
class SpatialAttn(nn.Module):   # based on channel and spatial feature calibration
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)
        return z

class MyFunc(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()
    @staticmethod
    def backward(ctx, gO):
        # Error during the backward pass
        raise RuntimeError("Some error in backward")
        return gO.clone()
def run_fn(a):
 out = MyFunc.apply(a)
 return out.sum()

class GCT(nn.Module):    # Gated Channel Transformation

    def __init__(self,num_channels=64, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.epsilon = epsilon 
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
         #print(" GCT") 
         #with autograd.detect_anomaly():
         #out = run_fn(x)
         #print(" out")
	 #out.backward() 
         #print("GCT - FORWARD")
         #print( "x ",x.shape)
         x_tmp = x
         x_orig = x
         num_channels = x.size(1)
         #print("num channel : ",num_channels)
         h = x.size(3)
         w = x.size(2)

         #print(" x -shape-    ",x.shape)

         self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1)).cuda()
         self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).cuda()
         self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).cuda()

         # parameter for spatial attention
         self.alpha_sp = nn.Parameter(torch.ones(1, num_channels, 1, 1)).cuda()
         self.gamma_sp = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).cuda()
         self.beta_sp = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).cuda()
         #self.epsilon = epsilon
         mode = self.mode
         after_relu = self.after_relu

         #print(" alpha ", self.alpha.shape)
         #print(" gamma ",self.gamma.shape)
         #print(" beta ",self.beta.shape)
         #exit()
         #print(" feature shape ", feature_map.shape)


         if self.mode == 'l2':

          # channel transformation and normaliztion
          print(" ---- Channel transformation -----")
          embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
          norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
          #print(" ---- channel transformation -----")
          #print(" embedding shape ",embedding.shape)
          #print(" norm            ",norm.shape)
          #print(" ---- --------------------- -----")

          print(" ---- Spatial transformation -----")
            
          embedding_spa = (x_tmp.pow(2).sum(1,keepdim =True) + self.epsilon).pow(0.5)*self.alpha_sp
          norm_spa = self.gamma_sp/(embedding_spa.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)



          x_tmp = x_tmp.mean(1, keepdim=True)  # e.g. 32,1,24,8
          h = x_tmp.size(2)
          w = x_tmp.size(3)
          x = x_tmp.view(x.size(0), -1)  # e.g. 32,192
          z = x_tmp
          for b in range(x_tmp.size(0)):
           z[b] /= torch.sum(z[b])

          z = z.view(x_tmp.size(0), 1, h, w)
          z = z * self.alpha_sp

          #print(" z - spatial ",z.shape)
          # return z



         elif self.mode == 'l1':
           if not self.after_relu:
              _x = torch.abs(x)
           else:
              _x = x
           embedding = _x.sum((2, 3), keepdim=True) * self.alpha
           norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
         else:
            print('Unknown mode!')
            #sys.exit()
            exit()
         ## compute spatial and channel attention 
         gate_chan = 1. + torch.tanh(embedding * norm + self.beta)
         gate_spa  = 1. + torch.tanh(embedding_spa*norm_spa + self.beta_sp)

         gate = gate_chan*gate_spa  ## spatial and channel attention fused together. 
         #gate = gate_spa 
         print (" gate channel :  ",gate_chan.shape)
         print(" gate spatial  : ",gate_spa.shape)

       
         #channel_transformed_feature = x_orig*gate_chan
         #spatially_transformed_feature = x_orig*z
         #spatially_transformed_feature = x_orig*gate_spa 

         #print(" channal wise transforemd feature ",gate_chan.shape," x ",x_orig.shape," = ",channel_transformed_feature.shape)
         #print(" spatial wise transforemd feature ",gate_spa.shape," x ",x_orig.shape," = ",spatially_transformed_feature.shape)
         #spatio_channel = gate_chan*z 
         #print(" channal wise transforemd feature ",gate.shape," x ",x_orig.shape," = ",channel_transformed_feature.shape)
         #print(" spatial wise transforemd feature ",z.shape," x ",x_orig.shape," = ",spatially_transformed_feature.shape)
         #spatio_channel = gate*z
         #print(" spatio_channle ",spatio_channel.shape)
         #x_tans = x_orig*spatio_channel
         #print(" channel + spatial transformed ",x_tans.shape)
         #transformed_feature = channel_transformed_feature + spatially_transformed_feature + x_orig
         #feature_concat = torch 
         #print(" channel + spatial transformed + original ",transformed_feature.shape)
         #exit()
         #print(" \n\n\n\ Exit  -------- Exit ") 
         #exit()
         #return transformed_feature
         #return x_orig * gate
         return gate 





__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True,num_features=0, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        for mo in self.base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1,1)

        self.num_features = num_features
        self.num_classes = 751 #num_classes to be changed according to dataset
        self.dropout = dropout
        out_planes = self.base.fc.in_features
        self.local_conv = nn.Conv2d(out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer1 = nn.Conv2d(256, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer2 = nn.Conv2d(512, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer3 = nn.Conv2d(1024, self.num_features, kernel_size=1,padding=0,bias=False)

        nn.init.kaiming_normal_(self.local_conv.weight, mode= 'fan_out')
#       init.constant_(self.local_conv.bias,0)
        self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
        init.constant_(self.feat_bn2d.weight,1) #initialize BN, may not be used
        init.constant_(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

        rand_tensor = torch.rand((2, 5,10))
        # self.SA1 = SpatialAttn()
        # self.SA2 = SpatialAttn()
        # self.SA3 = SpatialAttn()
        # self.SA4 = SpatialAttn()

        #self.GCT_l1 = GCT(rand_tensor)
        #self.GCT_l2 = GCT(rand_tensor)
        #self.GCT_l3 = GCT(rand_tensor)

        self.GCT1 = GCT(rand_tensor)
        self.GCT2 = GCT(rand_tensor)
        self.GCT3 = GCT(rand_tensor)
        #self.GCT4 = GCT(rand_tensor)

        # self.offset = ConvOffset2D(256)

##---------------------------stripe1----------------------------------------------#
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance0.weight, std=0.001)
        nn.init.constant_(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance1.weight, std=0.001)
        nn.init.constant_(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance2.weight, std=0.001)
        nn.init.constant_(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance3.weight, std=0.001)
        nn.init.constant_(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance4.weight, std=0.001)
        nn.init.constant_(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance5.weight, std=0.001)
        nn.init.constant_(self.instance5.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance6 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance6.weight, std=0.001)
        nn.init.constant_(self.instance6.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)  # FC Layer
        nn.init.normal_(self.instance_layer1.weight, std=0.001)
        nn.init.constant_(self.instance_layer1.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance_layer2 = nn.Linear(self.num_features, self.num_classes) # FC Layer
        nn.init.normal_(self.instance_layer2.weight, std=0.001)
        nn.init.constant_(self.instance_layer2.bias, 0)
##---------------------------stripe1----------------------------------------------#

##---------------------------stripe1----------------------------------------------#
        self.instance_layer3 = nn.Linear(self.num_features, self.num_classes) # FC Layer
        nn.init.normal_(self.instance_layer3.weight, std=0.001)
        nn.init.constant_(self.instance_layer3.bias, 0)
####  --------------------------------- GCT 33 ---------------------------------- #

        # self.instance_layer33 = nn.Linear(self.num_features, self.num_classes)  # FC Layer
        # nn.init.normal_(self.instance_layer33.weight, std=0.001)
        # nn.init.constant_(self.instance_layer33.bias, 0)

####  --------------------------------- GCT 33 ---------------------------------- #




        self.fusion_conv = nn.Conv1d(4, 1, kernel_size=1, bias=False)

        self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)    

            if name == 'layer1':

               #global x_layer1  
               x_attn1 = self.GCT1(x)
               x_layer1 = x*x_attn1 + x 
               #print(" x_layer_1 shape ",x_layer1.shape) 
               x = x*x_attn1 + x 

               continue
            if name == 'layer2':

                #global x_layer2
                x_attn2 = self.GCT2(x) 
                x_layer2 = x * x_attn2 + x 

                x = x*x_attn2 + x  

            if name == 'layer3':

                #global x_layer3
                x_attn3 = self.GCT3(x)
                x_layer3 = x*x_attn3 + x  

                x = x + x*x_attn3 

   
# Add parameter-free spatial attention before GAP
#         x_attn1 = self.SA1(x_layer1)
#         x_attn2 = self.SA2(x_layer2)
#         x_attn3 = self.SA3(x_layer3)
        #x_attn1 = self.GCT1(x_layer1)
        #x_attn2 = self.GCT2(x_layer2)
        #x_attn3 = self.GCT3(x_layer3)

        #x_attn33 = self.GCT1(x_layer3)

        #print()

        #x_layer1 = x_layer1*x_attn1 + x_layer1
        #x_layer2 = x_layer2*x_attn2 + x_layer2
        #x_layer3 = x_layer3*x_attn3 + x_layer3

        #x_layer33 = x_layer1*x_attn3

# Deep Supervision
        x_layer1 = F.avg_pool2d(x_layer1, kernel_size=(96, 32),stride=(1, 1))
        x_layer1 = self.local_conv_layer1(x_layer1)
        x_layer1 = x_layer1.contiguous().view(x_layer1.size(0), -1)
        x_layer1 = self.instance_layer1(x_layer1)

        x_layer2 = F.avg_pool2d(x_layer2, kernel_size=(48, 16), stride=(1, 1))
        x_layer2 = self.local_conv_layer2(x_layer2)
        x_layer2 = x_layer2.contiguous().view(x_layer2.size(0), -1)
        x_layer2 = self.instance_layer2(x_layer2)

        x_layer3 = F.avg_pool2d(x_layer3, kernel_size=(24, 8), stride=(1, 1))
        x_layer3 = self.local_conv_layer3(x_layer3)
        x_layer3 = x_layer3.contiguous().view(x_layer3.size(0), -1)
        x_layer3 = self.instance_layer3(x_layer3)

        # x_layer33 = F.avg_pool2d(x_layer33, kernel_size=(24, 8), stride=(1, 1))
        # x_layer33 = self.local_conv_layer33(x_layer33)
        # x_layer33 = x_layer33.contiguous().view(x_layer33.size(0), -1)
        # x_layer33 = self.instance_layer33(x_layer33)

        sx = int(x.size(2)/6)
        kx = int(x.size(2)-sx*5)  # change to int()
        x = F.avg_pool2d(x,kernel_size=(kx,x.size(3)),stride=(sx,x.size(3)))   # H4 W8

        out0 = x/x.norm(2,1).unsqueeze(1).expand_as(x) # use this feature vector to do distance measure
        # out0 = torch.cat([f3,out0],dim=1)
        x = self.drop(x)
        x = self.local_conv(x)
        x = self.feat_bn2d(x)
        x = F.relu(x) # relu for local_conv feature
        x6 = F.avg_pool2d(x, kernel_size=(6,1), stride=(1, 1))
        x6 = x6.contiguous().view(x6.size(0), -1)

        c6 = self.instance6(x6)

        x = x.chunk(6,2)
        x0 = x[0].contiguous().view(x[0].size(0),-1)
        x1 = x[1].contiguous().view(x[1].size(0),-1)
        x2 = x[2].contiguous().view(x[2].size(0),-1)
        x3 = x[3].contiguous().view(x[3].size(0),-1)
        x4 = x[4].contiguous().view(x[4].size(0),-1)
        x5 = x[5].contiguous().view(x[5].size(0),-1)

        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        c4 = self.instance4(x4)
        c5 = self.instance5(x5)
        return out0, (c0, c1, c2, c3, c4, c5,c6,x_layer1,x_layer2,x_layer3)


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

