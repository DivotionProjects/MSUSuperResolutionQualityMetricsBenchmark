import imageio
import shutil
from PIL import Image
from tqdm import tqdm
import time
import os

def check_yuv(path):
    return path[-3 : len(path)] == 'yuv'
def Fix(s):
    return (4 - len(s)) * '0' + s


def pyiqa_preprocess(metric_type, tmp_video_path, tmp_dir1_path, tmp_dir2_path, resolution, dist_video_path = None, gt_video_path = None, dist_frames_path = None, gt_frames_path = None):
    print("Preparing videos...")
    try:
        shutil.rmtree(tmp_dir1_path)
    except:
        pass
    try:
        shutil.rmtree(tmp_dir2_path)
    except:
        pass
    os.makedirs(tmp_dir1_path)
    os.makedirs(tmp_dir2_path)
    
    video_width, video_height = map(int, resolution.split('x'))
    
    is_yuv = False
    if metric_type == 'fr' and not(gt_video_path is None) and check_yuv(gt_video_path):
        is_yuv = True
    
    cnt = 0
    if dist_frames_path is None:
        dist_video = imageio.get_reader(dist_video_path, 'ffmpeg')
        for frame in dist_video:
            imageio.imwrite(os.path.join(tmp_dir1_path, 'smallframe' + Fix(str(cnt + 1)) + '.bmp'), frame)
             
            small_frame = Image.open(os.path.join(tmp_dir1_path, 'smallframe' + Fix(str(cnt + 1)) + '.bmp'))
            cur_frame = Image.new(small_frame.mode, (max(video_width, 224), max(video_height, 224)), '#000000')
            cur_frame.save(os.path.join(tmp_dir1_path, 'frame' + Fix(str(cnt + 1)) + '.bmp'))
            cnt += 1
    else:
        cnt = len(os.listdir(dist_frames_path))
        for i in range(cnt):
            small_frame = Image.open(os.path.join(dist_frames_path, sorted(os.listdir(dist_frames_path))[i]))
            small_frame.save(os.path.join(tmp_dir1_path, 'smallframe' + Fix(str(i + 1)) + '.bmp'))
            
            frame = Image.new(small_frame.mode, (max(video_width, 224), max(video_height, 224)), '#000000')
            frame.save(os.path.join(tmp_dir1_path, 'frame' + Fix(str(i + 1)) + '.bmp'))
    if metric_type == 'fr':
        if gt_frames_path is None:
            if is_yuv:
                try:
                    os.remove(tmp_video_path)
                except:
                    pass
                os.system(f"ffmpeg -y -s {resolution} -pixel_format yuv420p -i {gt_video_path} -vcodec rawvideo -pix_fmt bgr24 {tmp_video_path}")
                gt_video = imageio.get_reader(tmp_video_path, 'ffmpeg')
            elif not is_yuv:
                gt_video = imageio.get_reader(gt_video_path, 'ffmpeg')
                
            i = 0
            for frame in gt_video:
                imageio.imwrite(os.path.join(tmp_dir2_path, 'smallframe' + Fix(str(i + 1)) + '.bmp'), frame)

                small_frame = Image.open(os.path.join(tmp_dir2_path, 'smallframe' + Fix(str(i + 1)) + '.bmp'))

                cur_frame = Image.new(small_frame.mode, (max(video_width, 224), max(video_height, 224)), '#000000')
                cur_frame.save(os.path.join(tmp_dir2_path, 'frame' + Fix(str(i + 1)) + '.bmp'))
                i += 1
        else:
            for i in range(cnt):
                small_frame = Image.open(os.path.join(gt_frames_path, sorted(os.listdir(dist_frames_path))[i]))
                #small_frame = Image.open(os.path.join(gt_frames_path, 'frame' + Fix(str(i + 1)) + '.png'))
                small_frame.save(os.path.join(tmp_dir2_path, 'smallframe' + Fix(str(i + 1)) + '.bmp'))
                
                frame = Image.new(small_frame.mode, (max(video_width, 224), max(video_height, 224)), '#000000')
                frame.save(os.path.join(tmp_dir2_path, 'frame' + Fix(str(i + 1)) + '.bmp'))
    return cnt
    
            
# gt_Frames -> tmp_dir2
# dist_frames -> tmp_dir1
def calc_pyiqa_metric(model, metric_type, frames_cnt, dist_frames_path = None, gt_frames_path = None):
    score = 0.0
    full_time = 0.0
    
    for i in tqdm(range(frames_cnt)):
        dist_image = os.path.join(dist_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_dist_image = os.path.join(dist_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        gt_image = os.path.join(gt_frames_path, 'frame' + Fix(str(i + 1)) + '.bmp')
        small_gt_image = os.path.join(gt_frames_path, 'smallframe' + Fix(str(i + 1)) + '.bmp')
        
        try:
            if metric_type == 'nr':
                start_time = time.time()
                score += model(small_dist_image).item()
                start_time = time.time() - start_time
                full_time += start_time
            else:
                start_time = time.time()
                score += model(small_dist_image, small_gt_image).item()
                start_time = time.time() - start_time
                full_time += start_time
        except:
            if metric_type == 'nr':
                start_time = time.time()
                score += model(dist_image).item()
                start_time = time.time() - start_time
                full_time += start_time
            else:
                start_time = time.time()
                score += model(dist_image, gt_image).item()
                start_time = time.time() - start_time
                full_time += start_time
    
    score /= frames_cnt
    full_time /= frames_cnt
    
    return score, full_time
