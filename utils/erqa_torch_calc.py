import os
import json
import imageio as iio
from tqdm import tqdm
import imageio
import cv2
import time
import numpy as np
import torch
from utils.erqa_torch.erqa_torch import ERQA

def process_image(image: np.array, device):
    image = image.astype(np.float64) / 255 * 2 - 1
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float().to(device)

def Fix(s):
    return (4 - len(s)) * '0' + s

def get_erqa_torch_model():
    return ERQA()
def calc_erqa_torch_metric(model, device, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    score = 0.0
    full_time = 0.0

    device = torch.device(device)

    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        gt_image = os.path.join(gt_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_gt_image = os.path.join(gt_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        try:
            start_time = time.time()
            score += model(process_image(iio.imread(small_dist_image), device)[None, ...], process_image(iio.imread(small_gt_image), device)[None, ...], return_maps=False).item()
            start_time = time.time() - start_time
            full_time += start_time
        except:
            start_time = time.time()
            score += model(process_image(iio.imread(dist_image), device)[None, ...], process_image(iio.imread(gt_image), device)[None, ...], return_maps=False).item()
            start_time = time.time() - start_time
            full_time += start_time
        
    score /= frames_cnt
    full_time /= frames_cnt
        
    return score, full_time
