import argparse
import os
from train import train
from baseline import train_baseline
from metrics import get_metrics
from visualize import plot_roc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = "CASIA", 
                        choices = ["CASIA","Kinect"], 
                        help = "Specify dataset")
    parser.add_argument("--missm", type = str, default = "2D", 
                        choices = ["2D","3D"], 
                        help = "Missing Modality")
    parser.add_argument("--obj_model", type = str, default = "PointNet",
                        choices = ["PointNet", "EdgeConv"],
                        help = "3D model for choice")
    parser.add_argument("--attr", type = bool, default = True, 
                        help = "Include Face Attribute in Model")
    parser.add_argument("--aedl", type = bool, default = True,
                        help = "Choose to use AEDL or not")
    parser.add_argument("--num_classes", type = int, default = 52, 
                        choices = [52, 123], 
                        help = "Number of classes")
    parser.add_argument("--num_embeddings", type = int, default = 1024,
                        help = "Dimension of embeddings")
    parser.add_argument("--k", type = int, default = 20, 
                        help = "K-nn in EdgeConv")
    parser.add_argument("--beta", type = float, default = 2.0, 
                        help = "Focus Hyperparameter")
    parser.add_argument("--lamb1", type = float, default = 0.1, 
                        help = "Weight of AED Loss")
    parser.add_argument("--lamb2", type = float, default = 0.0001, 
                        help = "Weight of Reconstruction Loss")
    parser.add_argument("--input_size", type = int, default = 224,
                        help = "Input Size of 2D Image")
    parser.add_argument("--angle", type = int, default = 15,
                        help = "Rotation augmentation")
    parser.add_argument("--noise", type = int, default = 40,
                        help = "Gaussion noise augmentation")
    parser.add_argument("--crop", type = float, default = 0.1,
                        help = "Rotation augmentation")
    parser.add_argument("--img_mean", type = float, default = [0.485, 0.456, 0.406],
                        help = "Mean of Image Normalization")
    parser.add_argument("--img_std", type = float, default = [0.229, 0.224, 0.225],
                        help = "Standard Deviation of Image Normalization")
    parser.add_argument("--normalize", type = bool, default = True,
                        help = "Choose to normalize data or not")
    parser.add_argument("--data_dir", type = str, 
                        default = "/data/",
                        help = "Location of Raw Data")
    parser.add_argument("--output_dir", type = str,
                        default = "/output/",
                        help = "Location of Output")
    parser.add_argument("--batch_size", type = int, default = 32,
                        help = "Batch Size")
    parser.add_argument("--optimizer", type = str, default = "Adam",
                        choices = ["SGD","Adam","Adadelta"],
                        help = "Type of Optimizer")
    parser.add_argument("--lr", type = float, default = 0.001,
                        help = "Learning Rate")
    parser.add_argument("--betas", type = float, default = (0.9, 0.999),
                        help = "Optimizer Coefficient")
    parser.add_argument("--momentum", type = float, default = 0.9,
                        help = "SGD Momentum")
    parser.add_argument("--weight_decay", type = float, default = 0.001,
                        help = "Weight Decay")
    parser.add_argument("--decrease_lr", type = bool, default = True,
                        help = "Choose whether to decrease learning rate when loss not optimized")
    parser.add_argument("--step_size", type = int, default = 5,
                        help = "Number of epochs per learning rate decreasing")
    parser.add_argument("--valid", type = int, default = 1,
                        help = "Number of epochs per validation")
    parser.add_argument("--save", type = int, default = 5,
                        help = "Number of epochs per model saving")
    parser.add_argument("--gamma", type = float, default = 0.5,
                        help = "Percentage decreased each time")
    parser.add_argument("--num_epochs", type = int, default = 40,
                        help = "Number of Epochs for Training")
    parser.add_argument("--start_epoch", type = int, default = 0,
                        help = "Starting Epoch")
    parser.add_argument("--training_mode", type = str, default = "MTUT",
                        choices = ["MTUT", "baseline"])
    parser.add_argument("--test_mode", type = str, default = "metrics",
                        choices = ["metrics", "plots"])
    parser.add_argument("--train", type = bool, default = True,
                        help = "Turn on training mode")
    parser.add_argument("--pretrain", type = bool, default = False,
                        help = "Continue training or not")
    parser.add_argument("--model_path", type = str,
                        default = "",
                        help = "Load pre-trained model")
    args = parser.parse_args()
    train(args)