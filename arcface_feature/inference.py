import argparse
import os

import cv2
import numpy as np
import torch

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from backbones import get_model


from scipy.spatial.distance import cosine
import torch
import numpy as np
import cv2
import json

@torch.no_grad()
def batch_inference(weights, model_name, image_files, output_file):
    # 图像预处理
    imgs = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    imgs = torch.from_numpy(imgs)
    imgs.div_(255).sub_(0.5).div_(0.5)
    
    # 加载模型
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weights))
    net.eval()
    
    # 推理
    features = net(imgs).numpy()
    
    # 保存特征到JSONL文件
    with open(output_file, 'a') as f:
        for img_path, feat in zip(image_files, features):
            feature_dict = {"image_name": img_path, "feature": feat.tolist()}
            json_line = json.dumps(feature_dict)
            f.write(json_line + '\n')




def list_images_in_folders(parent_folder):
    # 存储所有文件夹中的图片路径列表
    all_images = []
    
    # 遍历父文件夹中的每一个子文件夹
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        
        # 确保它是一个文件夹
        if os.path.isdir(folder_path):
            # 存储当前文件夹中的图片路径
            images_in_current_folder = []
            
            # 遍历文件夹中的所有文件
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 构建文件的完整路径
                    file_path = os.path.join(folder_path, filename)
                    # 添加到当前文件夹的图片列表中
                    images_in_current_folder.append(file_path)
            
            # 将当前文件夹的图片列表添加到总列表中
            all_images.append(images_in_current_folder)
    
    return all_images

# 使用函数


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    # parser.add_argument('--network', type=str, default='r100', help='backbone network')
    # parser.add_argument('--weight', type=str, default='')
    # parser.add_argument('--img', type=str, default=None)
    # args = parser.parse_args()
    # "/workspace/arcface/verify_2000", "/workspace/arcface/model.pt", "r100"
    # inference(args.weight, args.network, args.img)

    parent_folder = '/workspace/comfyui_controlnet_aux/src/HQ_origin'  # 你的外层文件夹路径
    images_list = list_images_in_folders(parent_folder)
    for lst in images_list:
        batch_inference('/workspace/arcface/model.pt', 'r100', lst, 'output_features.jsonl')