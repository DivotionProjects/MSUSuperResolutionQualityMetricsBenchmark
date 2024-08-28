from utils.metrics_dict import MetricsInfo
from utils.beautify_dict import beautify_dict
from utils.vqmt_calc import calc_vqmt_metric
from utils.pyiqa_calc import calc_pyiqa_metric, pyiqa_preprocess
from utils.qalign_calc import calc_qalign_metric, get_qalign_model
from utils.dists_calc import calc_dists_metric, get_dists_model
from utils.lpips_calc import calc_lpips_metric, get_lpips_model
from utils.nima_calc import calc_nima_metric, get_nima_model
from utils.mdtvsfa_calc import calc_mdtvsfa_metric, get_mdtvsfa_model
from utils.linearity_calc import calc_linearity_metric, get_linearity_model
from utils.pieapp_calc import calc_pieapp_metric, get_pieapp_model
from utils.vsfa_calc import calc_vsfa_metric, get_vsfa_model
from utils.erqa_calc import calc_erqa_metric, get_erqa_model
#from utils.erqa_torch_calc import calc_erqa_torch_metric, get_erqa_torch_model
import pyiqa
import torch
import os
import tempfile
import imageio.v3 as iio

metrics_info = MetricsInfo()

tmp_dir = tempfile.gettempdir()
tmp_json1 = os.path.join(tmp_dir, 'tmp1.json')
tmp_json2 = os.path.join(tmp_dir, 'tmp2.json')
tmp_video = os.path.join(tmp_dir, 'tmp.mp4')
try:
    os.makedirs(os.path.join(tmp_dir, 'tmp1'))
except:
    pass
tmp_dir1 = os.path.join(tmp_dir, 'tmp1')
try:
    os.makedirs(os.path.join(tmp_dir, 'tmp2'))
except:
    pass
tmp_dir2 = os.path.join(tmp_dir, 'tmp2')

class Metric():
    def __init__(self, metric_name, device = None):
        self.metric_from = None
        self.metric_type = None
        self.full_metric_name = beautify_dict[metric_name]
        
        self.vqmt_metric_name = None
        self.vqmt_metric_type = None
        self.pyiqa_model = None
        
        self.last_frames_cnt = None
        self.device = None
        
        if metric_name in metrics_info.vqmt_names:
            if not (device is None) and \
                    ((not (metric_name in metrics_info.vqmt_devices.keys())) or \
                    (not (device in metrics_info.vqmt_devices[metric_name]))):
                raise ValueError(f'No device {device} for metric {self.full_metric_name} from list: {metrics_info.vqmt_devices[metric_name]}')
            
            self.metric_from = 'vqmt'
            self.metric_type = metrics_info.vqmt_types[metric_name]
            if not (device is None):
                self.vqmt_metric_name = metric_name + ' ' + device
            else:
                self.vqmt_metric_name = metric_name
            self.vqmt_metric_type = metrics_info.vqmt_metrics_vqmt_types[self.vqmt_metric_name]
        elif metric_name in metrics_info.pyiqa_names:
            self.metric_from = 'pyiqa'
            self.metric_type = metrics_info.pyiqa_types[metric_name]
            if metric_name == 'stlpips-alex':
                metric_name = 'stlpips'
            self.pyiqa_model = pyiqa.create_metric(metric_name, device=torch.device(device))
        elif 'q-align' in metric_name:
            self.metric_from = 'qalign'
            self.metric_type = metrics_info.qalign_types[metric_name]
            self.model = get_qalign_model(metric_name, device)
        elif metric_name == 'dists':
            self.metric_from = 'dists'
            self.metric_type = 'fr'
            self.model = get_dists_model()
        elif metric_name in ['lpips-vgg', 'lpips-alex']:
            self.metric_from = 'lpips'
            self.metric_type = 'fr'
            self.model = get_lpips_model(metric_name[6:])
        elif metric_name == 'nima':
            self.metric_from = 'nima'
            self.metric_type = 'nr'
            self.model = get_nima_model(device)
        elif metric_name == 'mdtvsfa':
            self.metric_from = 'mdtvsfa'
            self.metric_type = 'nr'
            self.model = get_mdtvsfa_model(device)
        elif metric_name == 'linearity':
            self.metric_from = 'linearity'
            self.metric_type = 'nr'
            self.model = get_linearity_model(device)
        elif metric_name == 'pieapp':
            self.metric_from = 'pieapp'
            self.metric_type = 'fr'
            self.model = get_pieapp_model(device)
        elif metric_name == 'vsfa':
            self.metric_from = 'vsfa'
            self.metric_type = 'nr'
            self.model = get_vsfa_model(device)
        elif metric_name in ['erqa1_0', 'erqa1_1', 'erqa2_0']:
            self.metric_from = 'erqa'
            self.metric_type = 'fr'
            self.model = get_erqa_model(metric_name[4:].replace('_', '.'))
        '''elif metric_name == 'erqa_torch':
            self.metric_from = 'erqa_torch'
            self.metric_type = 'fr'
            self.model = get_erqa_torch_model()
        '''
        self.device = device
            
    def predict(self, preprocessed=False, dist_video_path = None, gt_video_path = None, dist_frames_path = None, gt_frames_path = None):
        if self.metric_type == 'fr' and (gt_video_path is None and gt_frames_path is None):
            raise ValueError(f'Metric {self.full_metric_name} is Full-Reference, so it needs reference video')
        
        try:
            os.remove(tmp_json1)
        except:
            pass
        try:
            os.remove(tmp_json2)
        except:
            pass
        
        resolution = None
            
        if dist_video_path is None:
            frame_path = os.path.join(dist_frames_path, os.listdir(dist_frames_path)[0])
            frame = iio.imread(frame_path)
            h, w, c = frame.shape
            resolution = f'{w}x{h}'
        else:
            for frame in iio.imiter(dist_video_path):
                h, w, c = frame.shape
                resolution = f'{w}x{h}'
                break
        
        if self.metric_from == 'vqmt':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_vqmt_metric(self.vqmt_metric_name, tmp_dir2, tmp_dir1, resolution, self.vqmt_metric_type, tmp_json1, tmp_json2)
        elif self.metric_from == 'pyiqa':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_pyiqa_metric(self.pyiqa_model, self.metric_type, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'qalign':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_qalign_metric(self.model, self.metric_type, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'dists':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_dists_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'lpips':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_lpips_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'nima':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_nima_metric(self.model, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'mdtvsfa':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_mdtvsfa_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'linearity':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_linearity_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'pieapp':
            if not preprocessed:    
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_pieapp_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'vsfa':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_vsfa_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        elif self.metric_from == 'erqa':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_erqa_metric(self.model, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        '''elif self.metric_from == 'erqa_torch':
            if not preprocessed:
                self.last_frames_cnt = pyiqa_preprocess(self.metric_type, tmp_video, tmp_dir1, tmp_dir2, resolution, dist_video_path, gt_video_path, dist_frames_path, gt_frames_path)
            return calc_erqa_torch_metric(self.model, self.device, self.last_frames_cnt, tmp_dir1, tmp_dir2)
        '''
