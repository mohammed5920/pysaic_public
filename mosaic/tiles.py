from typing import Literal
import os
import math

from PIL import Image
from imohash import hashfile
import numpy as np
import cv2

import mosaic.tiles_funcs
import mosaic.tiles_vid
import util.misc
from pysaic import PySaic

class Tile:
    def __init__(self, path : str, size : int, source_type : Literal["vid", "pic"], tile_arr : np.ndarray[np.ndarray[np.uint8]], 
                 avg_colour : np.ndarray[np.uint8], tile_frame_idx : int = None , end_frame_idx : int = None):
        """initialise a new Tile object to store tile metadata and a low res version of that tile
        args:
        path: path to the vid/pic the tile comes from
        size: size of the low res version of the tile
        source_type: type of media the tile comes from
        tile_arr: low res array of the tile
        avg_colour: average colour of the tile
        tile_frame_idx: for tiles originating from videos, the frame index of the start of the tile
        end_frame_idx: for tiles originating from videos, the frame index of the end of the tile"""
        self.path : str = path
        self.size : int = size #since they're always square
        self.source_type = source_type
        self.arr : np.ndarray = np.rot90(np.fliplr(tile_arr))
        self.avg_colour = [colour.astype(np.int32) for colour in avg_colour]
        self.colour_key = self.avg_colour[0] << 16 | self.avg_colour[1] << 8 | self.avg_colour[2]
        self.frame_idx = tile_frame_idx or 0
        self.end_frame_idx = end_frame_idx or 1
        self.len = self.end_frame_idx - self.frame_idx

    def __len__(self):
        return self.len

    def as_fixed_res(self, res):
        """returns a reshaped version of the tile at (resxres)
        works even if the image is smaller than res"""
        return np.array(cv2.resize(self.arr, (res, res), interpolation=cv2.INTER_AREA), order="C")

def pic_tile(pic_path, ignore_pbar=False):
    """creates and returns a Tile object created from a picture"""
    try:
        if PySaic.stop_loading_flag: return 
        with open(pic_path, "rb") as raw_img:
            img = Image.open(raw_img).convert("RGB")
    except:
        util.misc.log(f"Error reading {pic_path}!")
        return
    finally:
        if not ignore_pbar: #ignore is used for testing 
            PySaic.pbar.update(1)
    dscale_factor = max(math.floor(np.min(np.divide(img.size, PySaic.settings.BASE_TILE_RES))), 1)
    img_arr = mosaic.tiles_funcs.crop(np.array(img)[::dscale_factor, ::dscale_factor])
    avg_colour = list(np.mean(img_arr, axis=(0, 1)).astype(np.uint8))
    return Tile(pic_path, img_arr.shape[0], "pic", img_arr, avg_colour)

def video_tiles(vid_path:str, ignore_pbar=False):
    """creates and returns a list of Tile objects created by going frame-by-frame 
    through a video and analysing the differences between frame n and frame n-1"""
    
    #hash video and check hash
    if PySaic.stop_loading_flag: return
    vid_hash = hashfile(vid_path, hexdigest=True)
    
    try:
        os.makedirs(f"caches/videos/{vid_hash}")
        with open(f"caches/videos/{vid_hash}/{os.path.basename(vid_path)}.txt", "w") as file: 
            pass #makes it easier to debug cache if i know what the original filename is
    except OSError:
        pass

    #read from frame and index cache
    if PySaic.stop_loading_flag: return
    if "unreadable.txt" in os.listdir(f"caches/videos/{vid_hash}") \
    or (data := mosaic.tiles_vid.cached_vid_tiles(vid_path, vid_hash)) is None: 
        if not ignore_pbar: PySaic.pbar.update(1)
        return
    frame_dict, good_frames, colours = data

    #create tile objects
    result : list[Tile] = []
    good_idxs = list(good_frames.keys()) + [sorted(frame_dict.keys())[-1]] #the last frame in the vid
    for i, (frame_idx, frame) in enumerate(good_frames.items()):
        if PySaic.stop_loading_flag: return

        avg_colour = np.array(colours[frame_idx], dtype=np.uint16)
        end_frame = good_idxs[i+1]
        #scanning the video helps find when the video cuts to a different colour
        #but fails when the video gradually changes colour
        #this can lead to hybrid mosaics slowly morphing into incorrect colours over time
        #so we go through the colours again, comparing the start of the tile (guaranteed good colour) to all frames up until the supposed end frame
        for j, colour in enumerate([colours[k] for k in range(frame_idx, end_frame)]):
            if mosaic.tiles_funcs.mse(avg_colour, np.array(colour, dtype=np.uint16)) > PySaic.settings.VID_TILE_DIFFERENCE_THRESHOLD:
                end_frame = j-1 + frame_idx
                break
        
        if end_frame <= frame_idx: continue #probably redundant but it's good to have peace of mind
        dscale_factor = math.floor(np.min(np.divide(frame.shape[0], PySaic.settings.BASE_TILE_RES)))
        if dscale_factor > 1: frame = frame[::dscale_factor, ::dscale_factor]

        result.append(Tile(vid_path, frame.shape[0], "vid", frame, avg_colour, frame_idx, end_frame))

    if not ignore_pbar: PySaic.pbar.update(1)
    return result