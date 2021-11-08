import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import csv
from torch.utils.data import DataLoader
from data import RGB3DObjDataset, Rescale, ToTensor, NormalizeImage
from autoencoder import CMAE
from aedl import AEDLoss


def train(args):
    if args.normalize:
        data_transforms = transforms.Compose([
            NormalizeImage(args.img_mean, args.img_std), 
            Rescale(args.input_size),
            ToTensor()
            ])
    else:
        data_transforms = transforms.Compose([Rescale(args.input_size),ToTensor()])
    
    train_dataset = RGB3DObjDataset(args.data_dir + "train/label.csv",
                                    args.data_dir + "train/RGB",
                                    args.data_dir + "train/3DObj",
                                    dataset = args.dataset,
                                    transform = data_transforms)
    
    valid_dataset = RGB3DObjDataset(args.data_dir + "val/label.csv",
                                    args.data_dir + "val/RGB",
                                    args.data_dir + "val/3DObj",
                                    dataset = args.dataset,
                                    transform = data_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.pretrain:
        ae = torch.load(args.model_path, map_location = device)
    else:
        ae = CMAE(args)
    
    if torch.cuda.device_count() > 1:
        ae = nn.DataParallel(ae)
    ae = ae.to(device)
    
    if args.optimizer == "Adam":
        optimizer = optim.Adam(ae.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(ae.parameters(), lr = args.lr, momentum = args.momentum)
    
    if args.decrease_lr:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size , gamma = args.gamma)
    
    criterion = AEDLoss(args)
    
    if not os.path.exists(args.output_dir + args.dataset + "_output/miss" + args.missm):
        os.mkdir(args.output_dir + args.dataset + "_output/miss" + args.missm)
    
    fid_train = open(args.output_dir + args.dataset + "_output/miss" + args.missm + "/loss_train.csv", 'w')
    writer_train = csv.writer(fid_train, lineterminator="\r\n")
    fid_train.close()

    fid_valid = open(args.output_dir + args.dataset + "_output/miss" + args.missm + "/loss_valid.csv", 'w') 
    writer_valid = csv.writer(fid_valid, lineterminator="\r\n")
    fid_valid.close()
    
    
    for epoch in range(args.start_epoch, args.num_epochs):
        print("----------")
        print("Epoch %d/%d" % (epoch + 1, args.num_epochs))
        ae.train()
        correct, total_loss, total_closs, total_div, total_lre = 0, 0, 0, 0, 0
        
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            label = data["label"].to(device)
            image = data["image"].to(device)
            pointcloud = data["pointcloud"].to(device)
            attribute = data["attr"].to(device)
            
            out, feat, recon = ae(image, pointcloud, attribute)
            inp = (image, pointcloud)
            loss, closs, div, lre = criterion(inp,label,out,feat,recon)
            loss.backward()
            optimizer.step()
            
            if args.decrease_lr:
                scheduler.step(epoch = epoch)
            
            if args.missm == "2D":
                pred = out[1].max(1)[1]
            else:
                pred = out[0].max(1)[1]
                
            correct += pred.eq(label).sum().item()
            total_loss += loss.item()
            total_closs += closs.item()
            total_div += div.item()
            total_lre += lre.item()
        
        acc = correct / len(train_dataset)
        total_loss = total_loss / len(train_dataloader)
        total_closs = total_closs / len(train_dataloader)
        total_div = total_div / len(train_dataloader)
        total_lre = total_lre / len(train_dataloader)
        
        print("Train Acc: %.4f Loss: %.4f Classification: %.4f Divergence: %.4f Reconstruction: %.4f" % (acc, total_loss, total_closs, total_div, total_lre))
        fid_train = open(args.output_dir + args.dataset + "_output/miss" + args.missm + "/loss_train.csv", 'a')
        writer_train = csv.writer(fid_train, lineterminator="\r\n")
        writer_train.writerow([epoch, acc, total_loss, total_closs, total_div, total_lre])
        fid_train.close()
        
        if epoch % args.valid == 0:
            ae.eval()
            correct, total_closs = 0, 0
            
            with torch.no_grad():
                for i, data in enumerate(valid_dataloader):
                    label = data["label"].to(device)
                    image = data["image"].to(device)
                    pointcloud = data["pointcloud"].to(device)
                    attribute = data["attr"].to(device)
            
                    out, feat, recon = ae(image, pointcloud, attribute)
                    inp = (image, pointcloud)
                    loss,closs,_,_ = criterion(inp,label,out,feat,recon)
            
                    if args.missm == "2D":
                        pred = out[1].max(1)[1]
                    else:
                        pred = out[0].max(1)[1]
                
                    correct += pred.eq(label).sum().item()
                    total_closs += closs.item()
        
                acc = correct / len(valid_dataset)
                total_closs = total_closs / len(valid_dataloader)
        
                print("Valid Acc: %.4f Classification Loss: %.4f" % (acc, total_closs))
                fid_valid = open(args.output_dir + args.dataset + "_output/miss" + args.missm + "/loss_valid.csv", 'a') 
                writer_valid = csv.writer(fid_valid, lineterminator="\r\n")
                writer_valid.writerow([epoch, acc, total_closs])
                fid_valid.close()
        
        if epoch % args.save == 0:
            torch.save(ae, args.output_dir + args.dataset + "_output/miss" + args.missm + "/ae_{:02d}.model".format(epoch))
    torch.save(ae, args.output_dir + args.dataset + "_output/miss" + args.missm + "/ae_final.model")
            
            
            
        
        
    
    
    
