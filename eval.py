import torch
torch.set_grad_enabled(False)

from utils import Metric, beautify_dict
import json
import os
import csv
from multiprocessing import Process

from functools import partial
import time



#Show Metrics' List
def f(i, args, vid, metric, model):

    #model = models[metric]
    score, time = model.predict(**args)
    print(score, time)
    with open(f'res/{i}.txt', 'w') as f:
        f.write(f'{vid}\n{metric}\n{score}\n{time}')
if __name__ == '__main__':
    print(list(beautify_dict.beautify_dict.keys()))
    
    metrics = ['mse', 'psnr', 'ne', 'erqa_torch', 'dbcnn' , 'clipiqa+', 'hyperiqa', 'lpips_vgg']
    models = {metric : Metric(metric, "cuda:0" if metric not in ('mse', 'psnr', 'ne') else None) for metric in metrics}

    
    
    

        
        
    with open('GTclusters.json') as json_data:
        l = json.load(json_data)
    proc = []
    root_path = r'X:\27a_bor\datasetextension'
    with open('eval.csv', 'a', newline='') as csvfile:
        fieldnames = ['vid', 'metric', 'score', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        i = 0
        writer.writeheader()
        for metric_data in l:
            gt_path = os.path.join(root_path, 'GTframes', metric_data['group'])
            for vid in metric_data["dist_videos"] + metric_data["compressed_videos"]:
                dist_path = os.path.join(root_path, vid)
                for metric in metrics:
                    print(vid)
                    print(metric)
                    print()
                    if os.path.exists(f'res/{i}.txt'):
                        i += 1
                        continue
                    if not os.path.exists(dist_path):
                        i += 1
                        continue
                    if not os.path.exists(gt_path):
                        i += 1
                        continue
                    if vid in metric_data["compressed_videos"]:
                        i += 1
                        continue
                    p = Process(target=f, args = (i, {'dist_frames_path':dist_path, 'gt_frames_path':gt_path}, vid, metric, models[metric]))
                    p.start()
                    proc.append(p)
                    while len(proc := [p for p in proc if p.is_alive()]) >= 8:
                        time.sleep(0.1)
                    #score, time = model.predict(dist_frames_path=dist_path, gt_frames_path=gt_path)
                    
                    
                    i += 1
    '''
    metric_name = ""
    device = "cuda:0"
    
    gt_path = "gt_video.mp4"
    sample_path = "sample_video.mp4"
    
    # No-Reference Metric
    estimator_nr = Metric("q-align_iqa", "cuda:0")
    score, time = estimator_nr.predict(dist_video_path=sample_path)
    print(f'Score of {estimator_nr.full_metric_name} is {score}')
    
    # Full-Reference Metric
    estimator_fr = Metric("topiq_fr", "cuda:1")
    score, time = estimator_fr.predict(dist_video_path=sample_path, gt_video_path=gt_path)
    print(f'Score of {estimator_fr.full_metric_name} is {score}')
    
    # Metric with path to Frames
    sample_frames_path = './frames/'
    score, time = estimator_nr.predict(dist_frames_path=sample_frames_path)
    print(f'Score of {estimator_fr.full_metric_name} is {score}')
    '''