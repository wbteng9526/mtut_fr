import torch
import torch.nn as nn

class AEDLoss(nn.Module):
    
    def __init__(self, args):
        super(AEDLoss, self).__init__()
        self.args = args
        self.beta = args.beta
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.missm = args.missm
        self.aedl = args.aedl
        self.cls_ = nn.CrossEntropyLoss()
        
    def forward(self, inp, label, out, feat, recon):
        div = torch.mean(torch.norm(feat[0] - feat[1], dim = 1))
        img_closs = self.cls_(out[0], label)
        obj_closs = self.cls_(out[1], label)
        
        if self.missm == "2D":
            diff = obj_closs - img_closs
            lre = torch.mean(torch.norm(recon[0] - inp[0])) + torch.mean(torch.norm(recon[1] - inp[0]))
            
            if self.aedl:
                if diff > 0:
                    loss = obj_closs + self.lamb2 * lre + self.lamb1 * (torch.exp(self.beta * diff) - 1) * div
                else:
                    loss = obj_closs + self.lamb2 * lre
            else:
                loss = obj_closs + self.lamb2 * lre + self.lamb1 * div
        
        else:
            diff = img_closs - obj_closs
            lre = torch.mean(torch.norm(recon[0] - inp[1])) + torch.mean(torch.norm(recon[1] - inp[1]))
            
            if self.aedl:
                if diff > 0:
                    loss = img_closs + self.lamb2 * lre + self.lamb1 * (torch.exp(self.beta * diff) - 1) * div
                else:
                    loss = img_closs + self.lamb2 * lre
            else:
                loss = img_closs + self.lamb2 * lre + self.lamb1 * div
        
        if self.missm == "2D":
            return loss, obj_closs, div, lre
        else:
            return loss, img_closs, div, lre