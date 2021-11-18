import torch
import torch.nn as nn
from model import PointNet, EdgeConv, initialize_model

class CMAE(nn.Module):
    def __init__(self, args):
        super(CMAE, self).__init__()
        self.args = args
        self.missm = args.missm
        self.attr = args.attr
        
        if args.model_name in ["resnet", "vgg", "squeezenet"]:
            self.img_encoder = initialize_model(model_name = args.model_name, num_classes = self.args.num_embeddings)
            
        self.img_fc = nn.Sequential(
            nn.Linear(self.args.num_embeddings, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.args.num_classes)
            )
        
        if args.obj_model == "PointNet":
            self.obj_encoder = PointNet(args)
        else:
            self.obj_encoder = EdgeConv(args)
            
        
        self.attr_encoder = nn.Sequential(nn.Linear(38, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        
            
        if self.missm == "2D":
            self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, 7, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64,  32,  4, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1, bias = False),
                    nn.Sigmoid()
                )
        else:
            self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(1024, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.ConvTranspose1d(128, 64, 1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.ConvTranspose1d(64, 3, 1),
                    nn.BatchNorm1d(3),
                    nn.ReLU()
                )
            
    def forward(self, img, obj, attr):
        batch_size = img.size(0)
        img_feat = self.img_encoder(img)
        img_out = self.img_fc(img_feat)
        obj_out, obj_feat = self.obj_encoder(obj)
        feat = (img_feat, obj_feat)
        
        if self.attr:
            attr_feat = self.attr_encoder(attr)
            img_feat = self.fc1(img_feat)
            obj_feat = self.fc1(obj_feat)
            img_feat = torch.cat((img_feat, attr_feat), dim = 1)
            obj_feat = torch.cat((obj_feat, attr_feat), dim = 1)
            img_feat = self.fc2(img_feat)
            obj_feat = self.fc2(obj_feat)
        
        if self.missm == "2D":
            img_feat = img_feat.view(-1, 256, 2, 2)
            obj_feat = obj_feat.view(-1, 256, 2, 2)
        else:
            img_feat = img_feat.view(-1, 1024)
            obj_feat = obj_feat.view(-1, 1024)
            img_rand = torch.randn(batch_size, 1024, 4000).cuda()
            obj_rand = torch.randn(batch_size, 1024, 4000).cuda()
            img_feat = img_feat.unsqueeze(2)
            obj_feat = obj_feat.unsqueeze(2)
            img_feat = torch.min(img_feat, img_rand)
            obj_feat = torch.min(obj_feat, obj_rand)
    
        img_recon = self.decoder(img_feat)
        obj_recon = self.decoder(obj_feat)
        
        out = (img_out, obj_out)
        recon = (img_recon, obj_recon)
        
        
        return out, feat, recon
        
