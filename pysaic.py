from typing import Literal
import math

from tqdm import tqdm

import pygame
import util.misc
import numpy as np
import ui.core

class Stage:
    """splits up the flow of the program into distinct 'stages' that can be activated and that have clear entry/exit points
    \n none of the class methods are designed to be executed directly by user code
    \n instead, flow is to be directed with PySaic.transfer_stage(), PySaic.switch_stage(), PySaic.return_stage()"""
    def __init__(self):
        self._return_func = None
    
    def start(self): 
        """executed when first entering the stage"""
        pass

    def update(self, events : list[pygame.Event]): 
        """executed every frame the stage is active"""
        pass

    def pause(self): 
        """executed when suspending the stage with the possibility of returning to it later with Stage.resume()"""
        pass
    def cleanup(self): 
        """executed when the stage cannot be returned to without re-initialising with Stage.start()"""
        pass

    def resume(self):
        """executed when returning from a suspended state (Stage.pause())""" 
        if self._return_func:
            self._return_func(self)
    
class Mosaic:
    """stores metadata regarding the mosaic settings, along with the required data required to match and render the mosaic itself"""
    def __init__(self):
        #attributes that cannot be directly set by user -----------------------------
        
        #set pre-analysis
        self.source = None #source object as defined in mosaic.source
        self.tiles = None #list of tile objects as defined in mosaic.tiles
        
        #set after analysis
        self.colours : np.ndarray = None #average colour of every tile
        self.clr_sq_sum : np.ndarray = None #np.sum(self.colours ** 2, axis=1) -> speeds up the matching algorithm by calculating this once
        
        #set after matching
        self.raw_matches : np.ndarray[int] | np.ndarray[np.ndarray[int]] = None #1d array of rgb values converted to list of tile indices (2d in case dithering is enabled)
        self.matches : np.ndarray[int] = None #same as m_list but always 1 dimensional (dithered choices are flattened)
        self.unique_tiles : set = None #set of all unique tiles used in the mosaic, used for the renderers

        #set for rendering
        self.tile_map : np.ndarray[int] = None
        self.tile_len_map: np.ndarray[int] = None
        self.mean_tile_len : int = None
        self.max_tile_len : int = None
        self.total_frames : int = None
        
        #everything else ---------------------------------------------------------------
        
        #pic - tiles don't move, source doesn't move
        #vid - tiles don't move, source moves
        #hyb - tiles move, source doesn't move (short for 'hybrid')
        self.mode : Literal["pic", "vid", "hyb"] = None

        self.source_path : str = None #path to source 
        self.source_scale : int = 1 #downscale level 
        self.tile_folder : str = None #path to folder with tiles

        self.enable_pics : bool = False #not controllable by user directly, evaluated according to every other setting
        self.enable_vids : bool = False #eg, media type in folder, media type valid for mode, media type toggled by user

        self.pic_tiles : list[str] = [] #path to all pic tiles
        self.vid_tiles : list[str] = [] #path to all video tiles

        self.rand_choice : int = 0 #dithering level
    
    def clone(self, orig):
        """copies over all user setting variables from one mosaic metadata object to the other"""
        for field, value in orig.__dict__.items():
            if field in ("mode", "source_path", "source_scale", "tile_folder", "enable_pics", 
                         "enable_vids", "pic_tiles", "vid_tiles", "rand_choice"):
                self.__dict__[field] = value
    
    def __eq__(self, value):
        for field, value in value.__dict__.items():
            if field in ("mode", "source_path", "source_scale", "tile_folder", "enable_pics", 
                         "enable_vids", "pic_tiles", "vid_tiles", "rand_choice"):
                if not self.__dict__[field] == value:
                    return False
        return True

class PySaic:
    settings = util.misc.Settings()
    settings.add("ANIM_SPEED", 1, lambda x : isinstance(x, (float, int)) and x > 0, 
                 "<float> scales the animation speed of the UI")
    settings.add("RAM_PERCENT", 75.0, lambda x : isinstance(x, (float, int)) and (0 < x <= 100),
                 "<float, 1-100> percentage of free RAM to use for the renderer")
    settings.add("CAP_TILE_LENGTHS", True, lambda x : isinstance(x, bool) or x in (0,1), 
                 "<bool> whether or not to cap tile lengths to the average tile length for the entire mosaic (for hybrid mode)")
    settings.add("DISPLAY", 0, lambda x : isinstance(x, int) and x >= 0, 
                 "<int> index of display to render to - check your display settings to find which number corresponds to which display")
    settings.add("STREAM_LENGTH", 1, lambda x : isinstance(x, int) and x >= 0, 
                 "<int> number of seconds of video to buffer ahead while in video mode")
    settings.add("VID_TILE_DIFFERENCE_THRESHOLD", 75, lambda x : isinstance(x, int) and x > 0,
                 "<int> difference between frame n and frame n-1 to be considered a new tile in a video")
    settings.add("ZOOM_STEPS", 30, lambda x : isinstance(x, int) and x > 0, 
                 "<int> number of discrete steps between each tile scale in the renderer (i.e how fast zooming happens)")
    settings.add("PREFETCH_MULTIPLIER", 4, lambda x : isinstance(x, int) and x >= 1 and not math.log2(x) % 1 ,
                 "<int, power of two> size multiplier of tiles to prefetch and then downscale from for streaming")
    settings.add("BASE_TILE_RES", 64, lambda x : isinstance(x, int) and x >= 1 and not math.log2(x) % 1, 
                 "<int, power of two> resolution a tile is downscaled to before analysing its average colour, saves RAM and processing power but affects quality/speed of rendering")

    resolution : tuple[int, int] = None
    UI : ui.core.UIInstance

    fps : int = None
    def change_fps(new):
        PySaic.fps = new
        PySaic.UI.settings.change_fps(new)

    stages : dict[str, Stage] = dict() 
    current_stage : Stage = None
    previous_stages : list[Stage] = []

    display_surf : pygame.Surface = None
    clock : pygame.Clock = None

    loading_thread : util.misc.ThreadWrapper = None
    stop_loading_flag = False

    vid_extensions = util.misc.ExtensionSet(['mkv', 'flv', 'vob', 'ogv', 'rrc', 'gifv', 'mng', 'mov', 'avi', 'qt', 
                'wmv', 'yuv', 'rm', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 
                'mpv', 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 
                'f4b', 'mod', "webm", "bik"])

    pic_extensions = util.misc.ExtensionSet(['bmp', 'gif', 'tiff', 'jpeg', 'jpg', 'mpeg', 'ico', 'bw', 'rgba', 'apng', 
                                   'dds', 'png', 'tif', 'webp'])

    temp = Mosaic() #unprocessed mosaic metadata
    mosaic = Mosaic() #processed mosaic metadata
    pbar : tqdm = None

    def switch_stage(stage_key : str):
        """switch to an entirely new stage, 
        ensuring all data from last stages are cleaned up 
        and no state is left lingering"""
        for stage in PySaic.previous_stages: stage.cleanup()
        PySaic.previous_stages = []
        if PySaic.current_stage: PySaic.current_stage.cleanup()
        PySaic.current_stage = PySaic.stages[stage_key]
        PySaic.current_stage.start()

    def transfer_stage(stage_key : str, return_func = None):
        """suspend a stage and run a new one, 
        keeping the last stage loaded in to return from
        \n additional argument: return_func (no arguments) ->
        a function that is executed once the stage being transferred from is returned to again"""
        PySaic.current_stage._return_func = return_func
        PySaic.previous_stages.append(PySaic.current_stage)
        PySaic.current_stage.pause()
        PySaic.current_stage = PySaic.stages[stage_key]
        PySaic.current_stage.start()

    def return_stage():
        """returns to the last suspended stage and executes its resume funcs
          if it exists else returns false"""
        if not PySaic.previous_stages: return False
        PySaic.current_stage.cleanup()
        PySaic.current_stage = PySaic.previous_stages.pop()
        PySaic.current_stage.resume()
        return True

if __name__ == "__main__":
    import main