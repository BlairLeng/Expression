from concurrent.futures import ThreadPoolExecutor
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from PIL import Image, ImageDraw
import numpy as np
import asyncio
import aiofiles
from io import BytesIO
import concurrent.futures
from open_pose import OpenposeDetector, draw_poses, PoseResult
import numpy as np
from typing import List, NamedTuple, Union
from glob import glob
import json
import cv2
from tqdm import tqdm

loss_map = {
    0: 16,
    1: 15,
    2: 14,
    3: 13,
    4: 12,
    5: 11,
    6: 10,
    7: 9,
    8: 8,
    9: 7,
    10: 6,
    11: 5,
    12: 4,
    13: 3,
    14: 2,
    15: 1,
    16: 0,
    17: 26,
    18: 25,
    19: 24,
    20: 23,
    21: 22,
    22: 21,
    23: 20,
    24: 19,
    25: 18,
    26: 17,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 35,
    32: 34,
    33: 33,
    34: 32,
    35: 31,
    36: 45,
    37: 44,
    38: 43,
    39: 42,
    40: 46,
    41: 47,
    42: 39,
    43: 38,
    44: 37,
    45: 36,
    46: 40,
    47: 41,
    48: 54,
    49: 53,
    50: 52,
    51: 51,
    52: 50,
    53: 49,
    54: 48,
    55: 59,
    56: 58,
    57: 57,
    58: 56,
    59: 55,
    60: 64,
    61: 63,
    62: 62,
    63: 61,
    64: 60,
    65: 67,
    66: 66,
    67: 65,
    68: 69,
    69: 68,
}

def openImage(img_path):
    image = Image.open(img_path)
    oriImg = np.array(image)
    return oriImg

def ensure_dir(file_path):
    """确保文件路径存在，如果不存在，则创建对应的目录"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def lsd_img(npImage, save_path, image_name):
    r_img = lsd(input_image = npImage)
    r_img = Image.fromarray(r_img)
    save_image_path = f'{save_path}/{image_name}.jpg'
    ensure_dir(save_image_path)
    r_img.save(save_image_path)
    return save_image_path

def getFaceKeypoints(poses):
    lst = []
    for point in poses.face:
        lst.append((point.x, point.y))
    return lst
# 假定其他必要的函数已经定义

def process_single_image(model, filepath, width, log_name):
    try:
        np_img = openImage(filepath)
        width, height = np_img.shape[:2]
        poses = getPoses(np_img, model)  # 注意：确保getPoses函数接受model作为参数
        keypoints_ori = getFaceKeypoints(poses)[:-1]

        np_flip = np_img[:, ::-1]
        poses = getPoses(np_flip, model)
        keypoints_flip = getFaceKeypoints(poses)[:-1]

        scaled_keypoints_origin = [(x * width, y * width) for x, y in keypoints_ori]
        scaled_keypoints_flipped = [(x * width, y * width) for x, y in keypoints_flip]
        flipped_scaled_keypoints_flipped = [(width - x - 1, y) for (x, y) in scaled_keypoints_flipped]

        temp_all_distance = 0
        if len(scaled_keypoints_origin) == len(scaled_keypoints_flipped) and len(scaled_keypoints_flipped) == 70:
            for i, v in enumerate(scaled_keypoints_origin):
                pointA_array = np.array(scaled_keypoints_origin[i])
                pointB_array = np.array(flipped_scaled_keypoints_flipped[loss_map[i]])
                distance = np.linalg.norm(pointA_array - pointB_array) / width
                temp_all_distance += distance

            # 记录到信息日志
            log_message = f"distance: {temp_all_distance} for {filepath} and width is {width} \n"
            with open(f"mirror_loss_info_log_{log_name}.txt", "a") as file:
                file.write(log_message)

            return {filepath: temp_all_distance}
        else:
            # 长度不匹配的错误记录
            error_message = f"error: length not right for {filepath} \n"
            with open(f"mirror_loss_error_log_{log_name}.txt", "a") as file:
                file.write(error_message)

            return {filepath: -999}
    except Exception as e:
        # 其他错误记录
        error_message = f"error: {e} for {filepath} \n"
        with open(f"mirror_loss_error_log_{log_name}.txt", "a") as file:
            file.write(error_message)

        return {filepath: -9999}


# 修改 getPoses 函数以接受一个模型实例作为参数
def getPoses(npImage, model):
    poses = model.detect_poses(npImage, include_hand=False, include_face=True)
    poses = poses[0]
    return poses

# 新函数：加载模型实例到 GPU
def load_models_on_gpu(model_count=24, gpu_device="cuda:2"):
    models = []
    for _ in range(model_count):
        model = OpenposeDetector.from_pretrained().to(gpu_device)
        models.append(model)
    return models

# 新函数：并行处理一个图像批次
def process_image_batch(model, image_paths, width, log_name):
    results = []
    for filepath in image_paths:
        result = process_single_image(model, filepath, width, log_name)  # 假设这是适配上述处理逻辑的函数
        results.append(result)
    return results

def process_mirror_loss(root_folder, width, gpu_device, log_name, model_count):
    # 加载模型实例
    models = load_models_on_gpu(model_count=model_count, gpu_device=gpu_device)

    # 收集所有JPEG图像文件路径
    all_image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_image_paths.append(os.path.join(root, file))
    
    # 将图像列表平均分配给每个模型
    total_models = len(models)
    images_per_model = len(all_image_paths) // total_models
    image_batches = [all_image_paths[i:i + images_per_model] for i in range(0, len(all_image_paths), images_per_model)]
    
    total_distance = []
    # 使用线程池并行处理每个图像批次
    with ThreadPoolExecutor(max_workers=total_models) as executor:
        futures = [executor.submit(process_image_batch, models[i], batch, width, log_name) for i, batch in enumerate(image_batches)]
        for future in concurrent.futures.as_completed(futures):
            batch_results = future.result()
            total_distance.extend(batch_results)

    # 保存处理结果
    with open(f'total_distance_{log_name}.json', 'w') as json_file:
        json.dump(total_distance, json_file)


if __name__ == "__main__":
    # 定义要处理的根目录和图像宽度
    root_folder = "/workspace/comfyui_controlnet_aux/src/HQ_origin"  # 替换为实际图像所在的目录路径
    # crawler_folder = "/workspace/SD_images_crawlers"
    width = 1024  # 或根据你的需要设置其他值

    # 调用主函数
    process_mirror_loss(root_folder, width, "cuda:2", "celebHQ", 10)
    # process_mirror_loss(crawler_folder, width, "cuda:3", "crawler")


