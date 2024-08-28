import os
import json

def is_yuv(path):
    if path is None:
        return False
    return path[-3 : len(path)] == 'yuv'
def run(cmd):
    return os.system(cmd)

global time_path
time_path = None

def get_time():
    with open(time_path, 'r') as f:
        timeinfo = json.load(f)
    return timeinfo["measure_time"] / timeinfo["frames"]

def calc_vqmt_metric(name, gtpath, distpath, resolution, mtype, tmp_json1, tmp_json2):
    global time_path
    time_path = tmp_json2
    score = 0.0
    res = tmp_json1
    time_res = tmp_json2
    isYUV = is_yuv(gtpath)
    cmd1 = f"vqmt -orig \"{gtpath}\" "
    if mtype == 0:
        cmd2 = f"-in \"{distpath}\" -metr {name} over YUV -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    elif mtype == 1:
        cmd2 = f"-in \"{distpath}\" -metr {name} over Y -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    elif mtype == 2:
        cmd2 = f"-in \"{distpath}\" -metr {name} over Y -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()

        cmd2 = f"-in \"{distpath}\" -metr {name} over U -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()

        cmd2 = f"-in \"{distpath}\" -metr {name} over V -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()
        score /= 3
        fulltime /= 3
    elif mtype == 3:
        mn, mp = name.split(' ')
        cmd2 = f"-in \"{distpath}\" -metr {mn} over YUV -dev {mp} -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    elif mtype == 4:
        mn, mp = name.split(' ')
        cmd2 = f"-in \"{distpath}\" -metr {mn} over Y -dev {mp} -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
        
        cmd2 = f"-in \"{distpath}\" -metr {mn} over U -dev {mp} -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()
        
        cmd2 = f"-in \"{distpath}\" -metr {mn} over V -dev {mp} -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()
        score /= 3
        fulltime /= 3
    elif mtype == 6:
        cmd = f"./vqmt -in \"{distpath}\" -metr {name} over Y -performance -json-file \"{res}\" > {time_path}"
        r = run(cmd)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    elif mtype == 7:
        cmd = f"./vqmt -in \"{distpath}\" -metr {name} over Y -performance -json-file \"{res}\" > {time_path}"
        r = run(cmd)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
        
        cmd = f"./vqmt -in \"{distpath}\" -metr {name} over U -performance -json-file \"{res}\" > {time_path}"
        r = run(cmd)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()
        
        cmd = f"./vqmt -in \"{distpath}\" -metr {name} over V -performance -json-file \"{res}\" > {time_path}"
        r = run(cmd)
        with open(res, 'r') as f:
            obj = json.load(f)
        score += obj['accumulated']['mean']['A']
        fulltime += get_time()
        
        score /= 3
        fulltime /= 3
    elif mtype == 8:
        mn, mp = name.split(' ')
        cmd = f"./vqmt -in \"{distpath}\" -metr {mn} over Y -dev {mp} -performance -json-file \"{res}\" > {time_path}"
        r = run(cmd)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    elif mtype == 9:
        m, c = name.split('_')
        cmd2 = f"-in \"{distpath}\" -metr {m} over {c} -performance -json-file \"{res}\" > {time_path}"
        if isYUV:
            r = run(cmd1 + f"{resolution} YUV420P " + cmd2)
        else:
            r = run(cmd1 + cmd2)
        with open(res, 'r') as f:
            obj = json.load(f)
        score = obj['accumulated']['mean']['A']
        fulltime = get_time()
    return score, fulltime
