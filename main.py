import importlib
import os
import sys
import time

import numba
import numpy as np
import pygame

import ui.core
import util.misc
from pysaic import PySaic

#close the splash image in the pyinstaller build
try:
    import pyi_splash  # type: ignore
    pyi_splash.close()
except ImportError:
    pass

#init fixed globals for other libraries
pygame.init()
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
sys.excepthook = util.misc.crash_handler
os.makedirs("caches/numba", exist_ok=True)
numba.config.CACHE_DIR = "caches/numba"
switch_to = ""

#parse settings file
if "settings.ini" in os.listdir():
    PySaic.settings.read()
    switch_to = "main"
else:
    switch_to = "main" #TODO fix compile file
if PySaic.settings.DISPLAY >= len(pygame.display.get_desktop_sizes()):
    util.misc.log("Display index is out of range, resetting to 0...")
    PySaic.settings.DISPLAY = 0

#set up pygame display according to settings file and display loading screen 
PySaic.resolution = np.divide(pygame.display.get_desktop_sizes()[PySaic.settings.DISPLAY], 1.2).astype(np.uint64)
PySaic.display_surf = pygame.display.set_mode(PySaic.resolution, pygame.RESIZABLE, display=PySaic.settings.DISPLAY, vsync=1)
pygame.display.set_caption(f"PySiac")

#set up PySaic global vars according to settings file
PySaic.fps = pygame.display.get_current_refresh_rate()
PySaic.clock = pygame.time.Clock()
PySaic.UI = ui.core.UIInstance(PySaic.resolution, PySaic.fps, 
                               display_surf=PySaic.display_surf, anim_speed=PySaic.settings.ANIM_SPEED, 
                               logger=util.misc.Logging)

#load in all the stages
try: 
    import stages  #type: ignore
    PySaic.stages = stages.load() #statically load in stages in pyinstaller build
except ImportError:
    for stage in [file for file in os.listdir() if "stage_" in file]: #dynamically load them (skips having to import all of them manually)
        class_name = stage[6:-3]
        PySaic.stages[class_name] = getattr(importlib.import_module(stage[:-3]), class_name.capitalize())()

PySaic.switch_stage(switch_to)

#main loop
while True:

    events = []
    for event in pygame.event.get():
        
        if event.type == pygame.QUIT: #save settings file and shutdown gracefully
            PySaic.stop_loading_flag = True
            PySaic.settings.save()
            pygame.quit()
            sys.exit(0)
        
        elif event.type == pygame.VIDEORESIZE: #handle window resizing and update the UI and any onscreen mosaics
            PySaic.resolution = np.array(event.size, dtype=np.int64)

            #recalculate mosaic if its become too small for the updated screen res 
            if (PySaic.current_stage == PySaic.stages["render"] or (PySaic.previous_stages and PySaic.previous_stages[0] == PySaic.stages["render"])) \
            and PySaic.mosaic.source.dscale_factor != np.ceil(np.max(np.divide(PySaic.mosaic.source.raw_size, PySaic.resolution))):

                if PySaic.loading_thread and PySaic.loading_thread.is_alive():
                    util.misc.log("Halting mosaic processing early to reload at a higher detail level...") 
                    PySaic.stop_loading_flag = True
                    PySaic.loading_thread.join()
                    PySaic.stop_loading_flag = False
                PySaic.switch_stage("mosaic")       

            PySaic.UI.resize(PySaic.resolution)
        else: 
            events.append(event)

    start = time.perf_counter()

    PySaic.current_stage.update(events)
    PySaic.UI.update(events, time.perf_counter() - start)

    PySaic.clock.tick(PySaic.fps)
    pygame.display.flip() #actually shows any changes to the display 

#pysaic flow

#main.py (initialises pygame window and global variables)
# |
# | (calls PySaic.switch_stage("main"))
# |
#stage_main.py (controls home screen for media selection and validation)
# |
#stage_mosaic.py (does tile analysis and mosaic processing for pic/hyb modes) ---(vid mode)---> starts mosaic.source.VidSource (dynamically streams in and processes video for vid mode)
# |
# | (at this point global mosaic data is processed and loaded)
# |
#stage_render.py (initialises a mosaic.render_classes.Renderer object and handles user control of the renderer)
# -----------------------------------------------------------------------------------------
# |                                                                                       |
#mosaic.render_classes.Renderer(draws the actual mosaic to the screen) ----------> mosaic.render_stream.StreamingManager(dynamically streams tiles from disk)
# |                                                                 ^                    |
# | (user hits save)                                                ---------------------                        
# |
#stage_save.py (saves the mosaic to disc and returns back to stage_render)