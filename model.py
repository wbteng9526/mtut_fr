import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torchvision import models
from facenet_pytorch import InceptionResnetV1

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.fc4 = nn.Linear(1024, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


def knn(x, k):
    inner = -2*torch.matmul(x, x.transpose(1,0))
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx.transpose(1,0) - inner - xx
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_edge_index(x,k=30):
    batch_size = x.size(0)
    num_points = x.size(1)
    n_dims = x.size(2)
  
    y = torch.empty((batch_size,num_points,k,n_dims*2),device="cuda")
    for i in range(batch_size):
        x1 = x[i]
        edge_index = knn(x1,k)
        edge = edge_index.view(-1)
        x2 = x1[edge,:]
        x2 = x2.view(-1,k,n_dims)
        x1 = x1.view(num_points,1,n_dims).repeat(1,k,1)
        y[i] = torch.cat((x2-x1,x1),dim=2)

    return y


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.args = args
        self.stn = STN3d()
        
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))
        
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, args.num_classes))
        
    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        feat = x.view(-1, 1024)
        
        x = self.fc1(feat)
        x = self.fc2(x)
        x = self.fc3(x)
        out = F.log_softmax(x, dim = 1)
        
        return out, feat
        


class EdgeConv(nn.Module):
    def __init__(self, args):
        super(EdgeConv, self).__init__()
        self.args = args
        self.k = args.k
        self.stn = STN3d()
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
    
        self.fc1 = nn.Sequential(nn.Linear(2048, 512, bias=False),nn.Dropout(0.5), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(nn.Linear(512, 256, bias=False),nn.Dropout(0.5), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.fc3 = nn.Linear(256, args.num_classes)
        
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)
        x = get_edge_index(x, self.k)
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = self.conv1(x)
        x1 = x.max(dim=-1,keepdim=False)[0]

        x = get_edge_index(x1.permute(0,2,1),self.k)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2(x)
        x2 = x.max(dim=-1,keepdim=False)[0]

        x = get_edge_index(x2.permute(0,2,1),self.k)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv3(x)
        x3 = x.max(dim=-1,keepdim=False)[0]
    
        x = get_edge_index(x3.permute(0,2,1),self.k)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv4(x)
        x4 = x.max(dim=-1,keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim = 1)
        
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        
        x = torch.cat((x1, x2), dim = 1)

        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        out = F.log_softmax(x, dim = 1)
        
        return out, x1


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    #input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        #input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        #input_size = 224
    
    elif model_name == "googlenet":
        """ GoogleNet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.fc = nn.Linear(1024, num_classes)
        #input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft





class CNN2D(nn.Module):
    
    def __init__(self, args):
        super(CNN2D, self).__init__()
        self.args = args

        self.main = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Flatten()
            )

        self.classifier = nn.Sequential(
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.Dropout(),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, args.num_classes)
            )

    def forward(self, x):
        feat = self.main(x)
        out = F.log_softmax(self.classifier(feat), dim = 1)

        return out, feat
    


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class DeepFace(nn.Module):
    
    def __init__(self, num_classes):
        super(DeepFace, self).__init__()
        # input size should be 3 * 152 * 152
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 11), nn.BatchNorm2d(32), nn.ReLU())
        self.pool1 = nn.MaxPool2d(3, 2, padding = 1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 16, 9), nn.BatchNorm2d(16), nn.ReLU())
        self.lc2d1 = nn.Sequential(LocallyConnected2d(16, 16, 55, 9, 1), nn.BatchNorm2d(16), nn.ReLU())
        self.lc2d2 = nn.Sequential(LocallyConnected2d(16, 16, 25, 7, 2), nn.BatchNorm2d(16), nn.ReLU())
        self.lc2d3 = nn.Sequential(LocallyConnected2d(16, 16, 21, 5, 1), nn.BatchNorm2d(16), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7056, 4096), nn.BatchNorm1d(4096), nn.ReLU())
        self.dp1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.lc2d1(x)
        x = self.lc2d2(x)
        x = self.lc2d3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.dp1(x)
        feat = self.fc2(x)
        x = F.log_softmax(feat, dim = 1)
        
        return feat

class VGGFace(nn.Module):
    def __init__(self, args):
        super(VGGFace, self).__init__()
        self.args = args
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, args.num_classes)
    
    def forward(self, x):
        # input size: 3 * 224 * 224
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out = F.log_softmax(x, dim = 1)
        
        return out

        


class DeepID(nn.Module):
    def __init__(self, num_classes):
        super(DeepID, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 20, 4),nn.BatchNorm2d(20),	nn.ReLU(),
									nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(20, 40, 3),nn.BatchNorm2d(40), nn.ReLU(),
									nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(40, 60, 3),nn.BatchNorm2d(60), nn.ReLU(),
									nn.MaxPool2d(2))
        self.deepID_layer = nn.Sequential(nn.Conv2d(60, 80, 2), nn.BatchNorm2d(80))
        self.dense_layer = nn.Linear(320, num_classes)
										
    def forward(self, x):
        # input size = 3 * 31 * 31
        x = self.block1(x)
        x = self.block2(x)
        x1 = self.block3(x)
        x2 = self.deepID_layer(x1)
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x = torch.cat((x1, x2), 1)
        feat = self.dense_layer(x)
        #out = F.log_softmax(feat, dim=1)
        
        return feat
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features
    

class DeepID2(nn.Module):
    def __init__(self, args):
        super(DeepID2, self).__init__()
        self.args = args
        self.block1 = nn.Sequential(nn.Conv2d(3, 20, 4),nn.BatchNorm2d(20),	nn.ReLU(),
									nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(20, 40, 3),nn.BatchNorm2d(40), nn.ReLU(),
									nn.MaxPool2d(2))
        self.block3 = nn.Sequential(LocallyConnected2d(40, 60, 8, 3, 1), nn.BatchNorm2d(60), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.deepID_layer = nn.Sequential(LocallyConnected2d(60, 80, 3, 2, 1), nn.BatchNorm2d(80))
        self.dense_layer1 = nn.Sequential(nn.Linear(1680, 160), nn.BatchNorm1d(160), nn.ReLU())
        self.dense_layer2 = nn.Linear(160, self.args.num_classes)
										
    def forward(self, x):
        # input size = 3 * 47 * 47
        x = self.block1(x)
        x = self.block2(x)
        x1 = self.block3(x)
        x2 = self.deepID_layer(x1)
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x = torch.cat((x1, x2), 1)
        feat = self.dense_layer1(x)
        out = self.dense_layer2(feat)
        out = F.log_softmax(out, dim=1)
        
        return out, feat
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features


class DeepID2Plus(nn.Module):
    def __init__(self, args):
        super(DeepID2Plus, self).__init__()
        self.args = args
        self.block1 = nn.Sequential(nn.Conv2d(3, 128, 4),nn.BatchNorm2d(128),	nn.ReLU(),
									nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(128, 128, 3),nn.BatchNorm2d(128), nn.ReLU(),
									nn.MaxPool2d(2))
        self.block3 = nn.Sequential(LocallyConnected2d(128, 128, 8, 3, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.deepID_layer = nn.Sequential(LocallyConnected2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128))
        self.dense_layer1 = nn.Sequential(nn.Linear(3200, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.dense_layer2 = nn.Linear(512, self.args.num_classes)
										
    def forward(self, x):
        # input size = 3 * 47 * 47
        x = self.block1(x)
        x = self.block2(x)
        x1 = self.block3(x)
        x2 = self.deepID_layer(x1)
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x = torch.cat((x1, x2), 1)
        feat = self.dense_layer1(x)
        out = self.dense_layer2(feat)
        out = F.log_softmax(out, dim=1)
        
        return out, feat
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features



def FaceNet(num_classes):
    # input size = 3 * 160 * 160
    return InceptionResnetV1(classify=True, num_classes = num_classes)