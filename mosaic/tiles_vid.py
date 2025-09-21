import subprocess
import json
import math
import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import numpy.typing as npt
import orjson
import cv2

if __name__ == "__main__":
    #hack to get debugging working
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

from pysaic import PySaic
import mosaic.tiles_funcs
import util.misc

def is_fixed_frame_rate(video_path):
    """returns True if fixed and False if variable/can't be determined"""
    def parse_fps(fps_str):
        num, den = map(int, fps_str.split('/'))
        return num / den if den != 0 else 0
    
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json", 
        video_path
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False #technically we can't tell if it's fixed frame rate if ffprobe can't read it?
    data = json.loads(output)
    r_fps = parse_fps(data["streams"][0]["r_frame_rate"])
    avg_fps = parse_fps(data["streams"][0]["avg_frame_rate"])
    return abs(r_fps - avg_fps) < 0.01

def scan_video(vid_path):
    """goes through a video and analyses the difference between each frame and the frame before it
    args:
    video_path: the path to the video
    
    returns a tuple:
    first item: dictionary of 'good frames' (frames that have a difference higher than PySaic.vid_tile_diff_thres) - {frame index : frame array}
    second item: dictionary of all frames in the video and their difference to the frame before it, sorted by highest difference
    third item: dictionary of all frames in the video and their average colour
    
    notes:
    \nraises LookUp error if cv2 fails to read a frame or if the video is shorter than 5 frames long
    \nthe function will always cache the first frame in a video"""
    good_frames = {} #frame idx : frame array
    frame_dict = {} #frame idx : difference to frame idx-1
    colours = {} #frame idx : avg colour

    vid_cap = cv2.VideoCapture(vid_path)
    vid_size = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    diff_scale_factor = max(math.floor(np.min(np.divide(vid_size, PySaic.settings.BASE_TILE_RES))), 1)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    if total_frames <= 3: raise LookupError

    for i in tqdm(range(total_frames), f"{vid_path} - scanning frames...", leave=False, mininterval=1):
        if PySaic.stop_loading_flag: return None, None, None
        #read from vid
        ret, frame = vid_cap.read()
        if not ret: raise LookupError
        
        #bgr->rgb and downscale frame
        frame = frame[:, :, ::-1] #bgr -> rgb
        resized_frame = frame[::diff_scale_factor, ::diff_scale_factor]
        cropped_frame = mosaic.tiles_funcs.crop(resized_frame, copy=False)
        colours[i] = mosaic.tiles_funcs.rgb_mean(cropped_frame) #calcualte average colour of downscaled frame
        
        #only count differences for frame > 0
        if i: 
            frame_dict[i] = diff = mosaic.tiles_funcs.mse(colours[i], colours[i-1]) #MSE of colours 
        else:
            frame_dict[i] = diff = 65025 #biggest diff possible (i think)
        
        #copy the cropped slice of the array so it doesnt keep the entire thing in memory
        if diff > PySaic.settings.VID_TILE_DIFFERENCE_THRESHOLD:
            good_frames[i] = cropped_frame.copy() 
    
    frame_dict = dict(sorted(frame_dict.items(), key=lambda x: x[1], reverse=True))
    colours = {k : [float(channel) for channel in v] for k, v in colours.items()} #so its json serialisable
    return frame_dict, good_frames, colours

def get_cached_indices(vid_hash):
    """returns the indices of cached frames found that are of a high enough resolution according to PySaic.base_tile_res"""
    cached_files = set([file for file in os.listdir(f"caches/videos/{vid_hash}") if file.endswith("jpg")])
    if not cached_files: return set()
    if min(Image.open(f"caches/videos/{vid_hash}/{cached_files.pop()}").size) < PySaic.settings.BASE_TILE_RES: return set()
    return set([int(filename.split(".")[0]) for filename in cached_files])

def extract_frame_from_cache(vid_hash, frame_idx):
    """returns np.array(Image.open({vid_hash/frame_idx.jpg}))"""
    with open(f"caches/videos/{vid_hash}/{frame_idx}.jpg", "rb") as cached_file: #faster than calling Image.open(path)
        big_array = np.array(Image.open(cached_file))
    dscale_factor = max(math.floor(np.min(np.divide(big_array.shape[:2], PySaic.settings.BASE_TILE_RES))), 1)
    if dscale_factor == 1:
        return big_array
    return big_array[::dscale_factor, ::dscale_factor]

def extract_frame_from_vid(vid_path, frame_idx, vid_cap : cv2.VideoCapture | None):
    """returns np array of frame extracted straight from the video
    args:
    video_path: the path to the video
    vid_hash: imohash.hashfile(video_path)
    frame_idx: frame to extract
    vid_cap: reference to the VideoCapture object used to seek in the video (or None if this is the first time the video is being accessed)

    returns a tuple:
    first item: np array of frame
    second item: VideoCapture object used to seek in the video (same as input if input is not None)

    raises LookupError if cv2 fails to read a frame
    """
    if vid_cap is None: 
        vid_cap = cv2.VideoCapture(vid_path) 
    vid_size = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dscale_factor = max(math.floor(np.min(np.divide(vid_size, PySaic.settings.BASE_TILE_RES))), 1)
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = vid_cap.read()
    if not ret: raise LookupError
    frame = mosaic.tiles_funcs.crop(frame[::dscale_factor, ::dscale_factor, ::-1])
    return frame, vid_cap

def cached_vid_tiles(vid_path, vid_hash) -> tuple[dict[int, npt.NDArray[np.uint8]], dict[int, float], dict[int, list[float]]] | None:
    """reads from vid tile cache if it exists and creates one if it doesn't\n
    args:
    video_path: the path to the video
    vid_hash: hexdigest of imohash.hashfile(video_path)
    
    returns a tuple:
    first item: dictionary of 'good frames' (frames that have a difference higher than PySaic.vid_tile_diff_thres) - {frame index : frame array}
    second item: dictionary of all frames in the video and their difference to the frame before it, sorted by highest difference
    third item: dictionary of all frames in the video and their average colour
    
    will return None if any errors occur during the reading phase"""

    frame_dict = good_frames = colours = vid_cap = None
    if PySaic.stop_loading_flag: return 
    
    try:
        #get cached metadata
        with open(f"caches/videos/{vid_hash}/indices.txt", "rb") as file:
            frame_dict = {int(k) : v for k,v in orjson.loads(file.read()).items()}
            good_frames = {i : None for i in sorted([idx for idx, diff in frame_dict.items() if diff > PySaic.settings.VID_TILE_DIFFERENCE_THRESHOLD])}
        with open(f"caches/videos/{vid_hash}/colours.txt", "rb") as file:
            colours = {int(k) : v for k,v in orjson.loads(file.read()).items()}
    
        #get cached frames
        cached_indices = get_cached_indices(vid_hash)
        for frame_idx in tqdm(good_frames, desc=f"{vid_path} - reading from cache...", leave=False, mininterval=1):
            if PySaic.stop_loading_flag: return
            if frame_idx in cached_indices:
                good_frames[frame_idx] = extract_frame_from_cache(vid_hash, frame_idx)
            else:
                try:
                    good_frames[frame_idx], vid_cap = extract_frame_from_vid(vid_path, frame_idx, vid_cap)
                    Image.fromarray(good_frames[frame_idx], mode="RGB").save(f"caches/videos/{vid_hash}/{frame_idx}.jpg")
                except LookupError:
                    with open(f"caches/videos/{vid_hash}/unreadable.txt", "w"): pass #creates empty file
                    util.misc.log(f"Error seeking in {vid_path}!")
                    return None
        return frame_dict, good_frames, colours

    except FileNotFoundError:
        if is_fixed_frame_rate(vid_path): #test if video has variable frame rate and skip it since they can't be seeked reliably
            try:
                frame_dict, good_frames, colours = scan_video(vid_path)
                if PySaic.stop_loading_flag: return

                with open(f"caches/videos/{vid_hash}/indices.txt", "wb") as cache:
                    cache.write(orjson.dumps(frame_dict, option=orjson.OPT_NON_STR_KEYS))
                with open(f"caches/videos/{vid_hash}/colours.txt", "wb") as cache:
                    cache.write(orjson.dumps(colours, option=orjson.OPT_NON_STR_KEYS))
                for frame_idx, frame_array in tqdm(good_frames.items(), desc=f"{vid_path} - saving to cache...", leave=False, mininterval=1):
                    Image.fromarray(frame_array, mode="RGB").save(f"caches/videos/{vid_hash}/{frame_idx}.jpg")
                return frame_dict, good_frames, colours
            
            except LookupError:
                with open(f"caches/videos/{vid_hash}/unreadable.txt", "w"): pass
                util.misc.log(f"Error reading {vid_path}!")

        else:
            with open(f"caches/videos/{vid_hash}/unreadable.txt", "w"): pass 
            util.misc.log(f"{vid_path} is variable frame rate!", print_out=False) 