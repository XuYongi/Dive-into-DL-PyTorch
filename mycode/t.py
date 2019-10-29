from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(".") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

transform = transforms.Compose([transforms.ToTensor()])

# mnist_train = torchvision.datasets.FashionMNIST(root='/home/wz209/xy/data/FashionMNIST/raw', train=True, download=True, transform=transform)
# mnist_test = torchvision.datasets.FashionMNIST(root='/home/wz209/xy/data/FashionMNIST/raw', train=False, transform=transform, download=True)
mnist_train = torchvision.datasets.FashionMNIST(root='./data', download=True,transform=transform,  train=False)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)