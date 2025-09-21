from typing import Literal

from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import pygame

import ui.util.exceptions
import ui.util.graphics

class FontWrapper(dict):
    """can store the UI font at multiple sizes that get scaled when the screen resizes"""
    def __init__(self, uii):
        self.uii = uii

    def __missing__(self, key):
        scaled_key = round(self.uii.scaler.min(key))
        self[key] = pygame.Font("assets/MuseoSans_700.otf", scaled_key)
        return self[key]
    
    def __getitem__(self, key) -> pygame.Font:
        return super().__getitem__(key)
    
    def resize(self):
        for key in self.keys():
            self.__missing__(key)

# -----------------------------------------------------------------------------------------

class MediaWrapper:
    def __init__(self, path, scale_type : Literal["fit", "fill"]=None, scale_res=None):
        self.path : str = path
        self.len : int = None
        self.max_len : int = None
        
        self.scale_type : Literal["fit", "fill"] = scale_type
        self.scale_res : tuple[int, int] = scale_res
        self.raw_size : tuple[int, int] = None
        self.size : tuple[int, int] = None
    
    def resize(self, scale_res):
        raise NotImplementedError

class PicWrapper(MediaWrapper):
    def __init__(self, path, scale_type : Literal["fit", "fill"]=None, scale_res=None):
        super().__init__(path, scale_type, scale_res)
        self.path = path
        self.media_type = "pic"
        try:
            self.pic = pygame.image.load(path).convert_alpha()
        except pygame.error:
            img = Image.open(path).convert("RGB")
            self.pic = pygame.image.frombytes(img.tobytes(), img.size, "RGB")
        self.len = 1
        if self.pic.size[0] * self.pic.size[1] >= int(1024 * 1024 * 1024 // 4 // 3): #pil will not load something like this but pygame will
            raise Image.DecompressionBombError
        if scale_type is not None:
            self.size = scale_res
            self.scaled_pic = ui.util.graphics.scale_surface(self.pic, scale_res, scale_type)
        else:
            self.size = self.pic.size
            self.scaled_pic = self.pic.copy()

    def resize(self, scale_res):
        self.scaled_pic = ui.util.graphics.scale_surface(self.pic, scale_res, self.scale_type)
        self.size = self.scale_res = scale_res

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx: raise KeyError
        return self.scaled_pic
    
class VideoWrapper(MediaWrapper):
    def __init__(self, path, scale_type : Literal["fit", "fill"]=None, scale_res=None):
        super().__init__(path, scale_type, scale_res)
        self.path = path
        self.scale_type = scale_type
        self.scale_res = scale_res
        self.media_type = "vid"
        
        self.fifo = list()
        self.frames = dict()
        
        self.vid_cap = cv2.VideoCapture(path)
        self.len = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
        if self.len <= 0:
            raise ValueError
        self.raw_size = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.scale_res is not None:
            self.size = self.scale_res
        else:
            self.size = self.raw_size

        self.max_len = int(10*1024*1024 / (self.size[0] * self.size[1] * 3))
    
    def resize(self, scale_res):
        self.fifo = []
        self.frames = {}
        self.size = self.scale_res = scale_res

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if idx > self.len:
            raise KeyError
        if idx in self.frames:
            return self.frames[idx]

        if not self.fifo or self.fifo[-1] != idx-1:
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        self.fifo.append(idx)
        if len(self.frames) > self.max_len:
            self.frames.pop(self.fifo.pop(0))

        ret, frame = self.vid_cap.read()
        if not ret: raise Exception()

        #fast downsampling pass to reduce the work by the smoothscaler later
        if self.scale_type and (scale_factor := int(np.min(np.divide(self.raw_size, self.size)))) > 1:
            frame = frame[::scale_factor, ::scale_factor]
        frame = pygame.surfarray.make_surface(np.rot90(np.fliplr(frame[:, :, ::-1]))).convert()
        
        if self.scale_type:
            frame = ui.util.graphics.scale_surface(frame, self.scale_res, self.scale_type)
        
        self.frames[idx] = frame
        return frame
    
def wrap_media(path, scale_type : Literal["fit", "fill"]=None, scale_res=None) -> MediaWrapper:
    try:
        return PicWrapper(path, scale_type, scale_res)
    except UnidentifiedImageError:
        return VideoWrapper(path, scale_type, scale_res)
    except Exception as e:
        raise ui.util.exceptions.UIMediaException(e)