import os
import math
import pandas as pd
import numpy as np
import torch
from skimage import io, transform
from augmentation import rotation
from imgaug import augmenters as iaa
from torch.utils.data import Dataset


class RGB3DObjDataset(Dataset):
    
    def __init__(self, csv_file, rgb_dir, obj_dir, dataset, transform=None):
        
        self.dataset = dataset
        self.label_frame = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.obj_dir = obj_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.label_frame)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.to_list()
            
        rgb_name = os.path.join(self.rgb_dir, self.label_frame.iloc[idx,0])
        rgb = io.imread(rgb_name)
        
        
        obj_name = os.path.join(self.obj_dir, self.label_frame.iloc[idx,1])
        if self.dataset == "CASIA":
            obj = TxtLoader(obj_name)
        else:
            obj = ObjLoader(obj_name)
        
        label = self.label_frame.iloc[idx,2]
        label = np.array(label)

        attr = self.label_frame.iloc[idx,3:]
        attr = np.array(attr, dtype=np.float64)
        
        sample = {'image':rgb, 'pointcloud':obj, 'label':label, 'attr':attr}
        
        if self.transform:
            sample = self.transform(sample)
        
            
        return sample


class Rescale(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pointcloud, label, attr = sample['image'], sample['pointcloud'], sample['label'], sample['attr']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'pointcloud': pointcloud,'label': label, 'attr': attr}


class NormalizeImage(object):
    
    def __init__(self,img_mean,img_std):
        self.img_mean = img_mean
        self.img_std = img_std
        
    def __call__(self, sample):
        image, pointcloud, label,attr = sample['image'], sample['pointcloud'], sample['label'], sample['attr']
        
        img_norm = (image - self.img_mean) / self.img_std
               
        return {'image': img_norm, 'pointcloud': pointcloud,'label': label,'attr':attr}


class ImageAugTransform(object):
    
    def __init__(self, angle, noise, crop):
        self.angle = angle
        self.noise = noise
        self.crop = crop
        self.aug = iaa.Sequential([
                iaa.Affine(rotate = [-1 * self.angle, self.angle]),
                iaa.AdditiveGaussianNoise(scale = self.noise),
                iaa.Crop(percent = self.crop)
            ])
        
    def __call__(self, sample):
        image, pointcloud, label,attr = sample['image'], sample['pointcloud'], sample['label'], sample['attr']
        
        image_aug = self.aug.augment_image(image)
        
        return {'image': image_aug, 'pointcloud': pointcloud,'label': label,'attr':attr}



class PointcloudAugTransform(object):
    
    def __init__(self, angle):
        self.direction_choice = [[1,0,0],[0,1,0],[0,0,1]]
        self.direction_index = np.random.randint(0,3)
        self.direction = self.direction_choice[self.direction_index]
        
        self.angle_choice = [-1 * angle, 0, angle]
        self.angle_index = np.random.randint(0,3)
        self.angle_radius = math.pi / 180 * self.angle_choice[self.angle_index]
    
    def __call__(self, sample):
        image, pointcloud, label,attr = sample['image'], sample['pointcloud'], sample['label'], sample['attr']
        
        pointcloud_aug = rotation(pointcloud, self.angle_radius, self.direction)
        
        return {'image': image, 'pointcloud': pointcloud_aug,'label': label,'attr':attr}


    
class ToTensor(object):

    def __call__(self, sample):
        image, pointcloud,label,attr = sample['image'], sample['pointcloud'], sample['label'], sample['attr']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        pointcloud = pointcloud.transpose((1, 0))
        return {'image': torch.from_numpy(image).float(),
                'pointcloud': torch.from_numpy(pointcloud).float(),
                'label': torch.from_numpy(label),
                'attr': torch.from_numpy(attr).float()}


def TxtLoader(filename):
    vertices = []
    
    f = open(filename)
    for line in f:
        v = line.split(",")
        v = [float(x) for x in v]
        vertices.append(v)
        
    f.close()
    
    return np.array(vertices)


def ObjLoader(fileName,swap_coord = True):
    vertices = []
        
    try:
        f = open(fileName)
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                
                if swap_coord == True:
                    # 0, 1, 2 ---> 2, 0, 1
                    vertex = [float(line[index3:]), float(line[index1:index2]), float(line[index2:index3])]                   
                    
                else:
                    vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:])]                   
                
                vertices.append(vertex)
        f.close()
        
        return np.array(vertices)
            
    except IOError:
        print(".obj file not found")
        
