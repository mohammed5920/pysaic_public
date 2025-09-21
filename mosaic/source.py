from concurrent.futures import ThreadPoolExecutor
import math
import os
import time

from PIL import Image
import cv2
import numpy as np

from pysaic import PySaic
import mosaic.match
import mosaic.render_funcs
import util.misc
import ui.util.exceptions

class Source:
    def __init__(self, size, dscale_factor, path):
        self.raw_size : tuple[int, int] = size #raw image (without scaling)
        self.dscale_factor : int = dscale_factor #how much it's been downscaled by
        self.scaled_size : tuple[int, int] = tuple(np.divide(self.raw_size, dscale_factor).astype(np.int64))
        self.path : str = path #path to the source 

class PicSource(Source):
    """object to hold pic source metadata in\n
    scales it to fit to screen and stores the size, its path and the rgb pixels within"""
    def __init__(self, photo_path : str, max_res : tuple[int, int]):
        #blend transparency into the image before converting to rgb
        image = Image.open(photo_path).convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, (0,0,0,255)), image).convert("RGB")
        self.raw_size : tuple[int, int] = image.size #raw image (without scaling)

        dscale_factor = math.ceil(np.max(np.divide(image.size, max_res))) #downscale the image to fit the max_res
        if dscale_factor > 1:
            image = image.resize((image.size[0] // dscale_factor, image.size[1] // dscale_factor))
        blocks = np.array(image).reshape(-1, 3).astype(np.uint8) #reshape to be 2d instead of 3d (just a long strip of colours)

        self.dscale_factor : int = dscale_factor #how much it's been downscaled by
        self.scaled_size : tuple[int, int] = tuple(np.divide(self.raw_size, dscale_factor).astype(np.int64))
        self.path : str = photo_path #path to the source 
        self.blocks = blocks

class VidSource(Source):
    """object to store vid source metadata in
    and to also dynamically stream and match as the video plays\n
    scales it to fit to screen and stores the size and its path,
    and streams in its raw RGB pixels and matches it automatically on separate threads.\n
    frames can be accessed like a cv2.VidCapture object by seek()ing to the start frame
    and then calling read_frame() to get the frame_pos+1'nd frame after that"""
    def __init__(self, video_path : str, max_res : tuple[int, int]):
        self.vid_cap = cv2.VideoCapture(video_path)
        size = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dscale_factor = math.ceil(np.max(np.divide(size, max_res)))

        self.path : str = video_path
        self.raw_size : tuple[int, int] = size
        self.dscale_factor : int = dscale_factor
        self.scaled_size : tuple[int, int] = tuple(np.divide(self.raw_size, dscale_factor).astype(np.int64))

        self.frame_pos = 0
        self.frame_count = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
        self.unique_tiles = set()
        self.cmap = np.zeros(2**24, dtype=np.int32)
        self.cmap.fill(-1)

        #you seek to a given frame position with seek()
        #and then use read_frame() to increment the frame position and get the next frame after that
        self.kill_threads_flag = False
        self.stream_buffer = {} #frame idx : reshaped raw video frame
        self.streaming_thread : util.misc.ThreadWrapper = None
        self.processed_buffer = {} #frame idx : matched frame array
        self.processing_thread : util.misc.ThreadWrapper = None

    def _stream_thread(self):
        vid_pos = self.frame_pos
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, vid_pos)
        while not self.kill_threads_flag:
            if self.kill_threads_flag: return
            if vid_pos == self.frame_count \
            or vid_pos >= self.frame_pos + PySaic.settings.STREAM_LENGTH*self.fps:
                time.sleep(0.001) 
                continue

            ret, frame = self.vid_cap.read()
            if not ret: raise ui.util.exceptions.UIMediaException("Error streaming source video!")
            scaled_frame = frame[::self.dscale_factor, ::self.dscale_factor, ::-1] #downscale video to fit
            resized_frame = scaled_frame[:frame.shape[0] // self.dscale_factor, :frame.shape[1] // self.dscale_factor] #crop the rest
            self.stream_buffer[vid_pos] = resized_frame.copy().reshape(-1, 3).astype(np.uint8)
            vid_pos += 1

    def _process_thread(self):
        while not self.kill_threads_flag:
            if not self.stream_buffer: 
                time.sleep(0.001)
                continue
            
            for idx in list(self.stream_buffer.keys()):
                if self.kill_threads_flag: break
                raw_frame = self.stream_buffer.pop(idx)
                frame, new = mosaic.match.build_matches_st(PySaic.mosaic.colours, PySaic.mosaic.clr_sq_sum, raw_frame, self.cmap, 12)
                
                #mosaic.match.build_matches_st returns the matched array, and a set of unique colours discovered in that array
                #this is NOT like how regular matching works, where it finds the unique tiles used in that frame
                #these are JUST the new ones that havent been seen before, so they need to be accumulated over the source of the streaming process
                self.unique_tiles.update(new)

                frame = np.reshape(frame[:self.scaled_size[1]*self.scaled_size[0]], (self.scaled_size[1], self.scaled_size[0]))
                self.processed_buffer[idx] = frame

    def kill_threads(self):
        """kills all background processing threads"""
        while ((self.processing_thread and self.processing_thread.is_alive()) 
        or (self.streaming_thread and self.streaming_thread.is_alive())):
            self.kill_threads_flag = True
            time.sleep(1/PySaic.fps)
        self.kill_threads_flag = False
        if self.streaming_thread: self.streaming_thread.join() #get any exceptions
        if self.processing_thread: self.processing_thread.join() #get any exceptions

    def seek(self, new_pos : int): #
        """used to initialise the thread system and/or reset it to a new position
        \nargs:
        new_pos: frame to seek to"""
        self.kill_threads()
        print(f"Seeking to {new_pos}...")
        self.frame_pos = new_pos
        self.stream_buffer, self.processed_buffer = {}, {}
        self.streaming_thread = util.misc.ThreadWrapper(target=self._stream_thread)
        self.processing_thread = util.misc.ThreadWrapper(target=self._process_thread)
        self.streaming_thread.daemon = self.processing_thread.daemon = True
        self.streaming_thread.start(), self.processing_thread.start()

    def reset(self):
        """kills the threads and resets all buffers"""
        self.kill_threads()
        self.unique_tiles = set()
        self.cmap.fill(-1)

    def check_frame(self, frame_idx):
        """check whether frame_idx exists in the buffers
        and whether the streaming has crashed"""

        if not self.streaming_thread.is_alive(): 
            self.streaming_thread.join()
        if not self.processing_thread.is_alive(): 
            self.processing_thread.join()

        if frame_idx in self.processed_buffer:
            return True
        else: 
            return False
       
    def read_frame(self, hold=False) -> np.ndarray:
        """get the next frame
        args:
        hold: block the processing thread until the frame is ready - return False otherwise"""
        while True:
            if self.check_frame(self.frame_pos):
                self.processed_buffer.pop(self.frame_pos-1, None)
                self.frame_pos += 1
                return self.processed_buffer[self.frame_pos-1]
            if not hold:
                return False
            time.sleep(1/self.fps)