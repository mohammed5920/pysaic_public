import math
import time

import numpy as np
import pygame

from pysaic import PySaic
import mosaic.render_funcs
import mosaic.render_stream
import ui.base
import ui.util
import ui.util.graphics

class Renderer(ui.base.Component):
    def __repr__(self):
        return f"Renderer ({PySaic.mosaic.mode})"
    def __init__(self, ui_instance, **kwargs):
        super().__init__(ui_instance, **kwargs)
        #component initialision 
        self.frame_pos = 0
        self.sm = mosaic.render_stream.StreamingManager()

        #read only references to the main mosaic
        self.source = PySaic.mosaic.source
        self.tiles = PySaic.mosaic.tiles
        self.unique_tiles = PySaic.mosaic.unique_tiles

        #mosaic metadata
        if PySaic.mosaic.mode == "vid": 
            self.source.seek(0)
            #processed_match_array, this is what is actually rendered to the screen
            self.pm_arr = self.source.read_frame(hold=True)
            self.fps = self.source.fps
        else: 
            self.pm_arr = np.reshape(PySaic.mosaic.matches, (self.source.scaled_size[1], self.source.scaled_size[0])) #read the processed mosaic from global memory
            self.fps = self.uii.settings.FPS
        self.pm_arr_cropped = None #used for debugging and streaming
    
        #mouse movement vars
        self.click_pos = None #position of the mouse at frame n-1 as long as the renderer is clicked
        self.x_offset, self.y_offset = np.divide(self.source.scaled_size, 2) #centre of the frame

        self.last_frame = None #used to prevent repainting a frame that doesnt change, store the last rendered frame
        self.lf_delay = 0 #no of frames to repaint before returning self.last_frame
        
        self.tile_size = 0 #size of a tile - increases in powers of two
        self.zoom = 0 #ranges from 0 - zoom_steps where it interpolates between different tile sizes
        self.zoom_steps = PySaic.settings.ZOOM_STEPS #essentially how fast it zooms 
        self.min_size = np.min(self.uii.scaler.display_res)
        self.resize((1,1))
        self.max_size = self.sm.res_limit
        self.tile_size = self.min_size

        #vid playback vars
        self.is_playing = False 
        self.is_buffering = False
        self.elapsed = 0 
        self.streamed = 0
        
        #streaming
        self.debug_streaming = False
        
        if PySaic.mosaic.mode == "hyb":
            print("Window may lag while videos are being streamed in, hitting ESCAPE will pause streaming.")
            print("If the renderer becomes responsive but takes too long loading the entire image,")
            print("zooming in will reset the streaming to only focus on the zoomed portion of the image.")
        self.sm.stream(self.unique_tiles, self.min_size)

    def while_clicked(self, translated_mouse_coords):
        zoom = self.tile_size * (1+(self.zoom/self.zoom_steps))
        if self.click_pos is not None:
            new_offset = np.subtract(translated_mouse_coords, self.click_pos)
            if np.any(new_offset):
                self.last_frame = None
                self.x_offset -= (new_offset[0] / zoom)
                self.y_offset -= (new_offset[1] / zoom)
        self.click_pos = translated_mouse_coords

    def on_up(self):
        self.click_pos = None

    def resize(self, xy):
        if not self.is_alive: return
        self.sm.halt()
        self.size = np.array(self.uii.scaler.display_res, dtype=np.int64) #set to fullscreen
        min_size = None
        
        ts = 1
        while True: #im sure theres some sort of math oneliner that would find this but this is easier to read
            _ = np.multiply(self.source.scaled_size, ts)
            if not np.all(_ <= self.size):
                min_size = int(ts // 2) #find the largest tile size that would fit the entire mosaic on screen
                break
            ts *= 2
        if min_size is None: 
            raise Exception("Mosaic is too big.")
        
        if min_size > self.tile_size:
            self.tile_size = min_size
            self.zoom = 0
        self.min_size = min_size
        
        self.sm.partition(self.min_size, self.size)
        if self.tile_size > self.sm.res_limit:
            self.tile_size = self.sm.res_limit
            self.zoom = self.zoom_steps - 1
        self.max_size = self.sm.res_limit
        
        if self.tile_size == self.min_size:
            self.sm.stream(self.unique_tiles, self.min_size)
        self.last_frame = None

    def get_surf(self, _): #modeled after windows Photos app
        
        # ----------------------------------------------------------------------- start of frame
        zoom = self.tile_size * (1+(self.zoom/self.zoom_steps))
        res = np.divide(self.size, zoom) #number of tiles to draw
        res = min(res[0], self.source.scaled_size[0]), min(res[1], self.source.scaled_size[1]) #limit to size of source
        res = math.ceil(res[0]), math.ceil(res[1]) 
        debug = [f"Zoom: {zoom}", f"Tile size/inter zoom: {self.tile_size}/{self.zoom}", f"Min/fast/max sizes: {self.min_size}/{self.sm.fast_limit}/{self.max_size}"]
        overscan = 1
        result = pygame.Surface((min(res[0]*zoom, self.size[0]), min(res[1]*zoom, self.size[1])))
        if not self.is_alive:
            return result #prevent it from drawing anything when it's being deleted to prevent errors

        if PySaic.mosaic.mode != "pic":
            debug.append(f"Playing? - {self.is_playing}")
            debug.append(f"Current frame pos: {self.frame_pos}")
        if PySaic.mosaic.mode == "vid":
            debug.append(f"Streamed: {len(self.source.stream_buffer)}")
            debug.append(f"Processed: {(1-len(self.source.stream_buffer)/(PySaic.settings.STREAM_LENGTH*self.source.fps))*100:.2f}%")
            if self.elapsed:
                debug.append(f"Process rate: {self.streamed/self.elapsed*100:02f}% - {self.streamed/self.elapsed*self.source.fps:.2f} FPS")

        if self.is_playing: 
            self.elapsed += 1
        else: 
            self.elapsed = 0
            self.streamed = 0

        #to avoid CPU turning into hot potato, cache last frame and return it 
        #if nothing has changed (not moving and not being moved)
        if self.last_frame and not self.sm.bg_thread.is_alive() \
        and (PySaic.mosaic.mode == "pic" or (not self.is_playing or self.is_buffering)): 
            self.debug_strings = debug
            self.lf_delay += 1
            if self.lf_delay > 2: return self.last_frame 
        else:
            self.streamed += 1
            self.lf_delay = 0
                
        # ----------------------------------------------------------------------- zooming and mouse movement
        buffer = pygame.Surface(np.multiply(res, self.tile_size)) 
        #correct for zooming in, base offset - no. of tiles being rendered / 2
        x_off = self.x_offset - res[0] / 2
        y_off = self.y_offset - res[1] / 2
        #bounds checking
        bx_off = max(0, min(x_off, self.source.scaled_size[0] - res[0]))
        by_off = max(0, min(y_off, self.source.scaled_size[1] - res[1]))
        #correct stored offsets
        self.x_offset -= min(0, x_off) + max(0, x_off - (self.source.scaled_size[0] - res[0]))
        self.y_offset -= min(0, y_off) + max(0, y_off - (self.source.scaled_size[1] - res[1]))
        debug.append(f"Stored x/y: {self.x_offset}/{self.y_offset}")
        debug.append(f"Adjusted x/y: {bx_off}/{by_off}")
        #add overscan
        if self.tile_size > 1: 
            res = np.add(res, overscan)
        debug.append(f"Tile grid resolution: ({res[0]}, {res[1]})")

        # ----------------------------------------------------------------------- rendering

        start = time.perf_counter()
        self.pm_arr_cropped = self.pm_arr[
            int(by_off) : int(by_off)+res[1],
            int(bx_off) : int(bx_off)+res[0]
        ]
        with self.sm.get_store(self.tile_size) as tile_store:
            render = mosaic.render_funcs.draw(int(bx_off), res[0], int(by_off), res[1], PySaic.mosaic.tile_map, 
                        tile_store.raw, self.pm_arr, self.tile_size, #pass in pm_arr instead of pm_arr_cropped for an extra layer of bounds checking
                        PySaic.mosaic.mode == "hyb", self.frame_pos, 
                        PySaic.mosaic.tile_len_map, self.sm.bg_thread.is_alive() and not self.sm.iterating)
        debug.append(f"Render time: {(time.perf_counter() - start)*1000:.2f}")

        #adjust for subtile movement
        if self.tile_size > 1:
            os_x = int(bx_off%1*self.tile_size)
            os_y = int(by_off%1*self.tile_size)
            render = render[os_x : os_x + (res[0]-overscan)*self.tile_size, os_y : os_y + (res[1]-overscan)*self.tile_size]
            debug.append(f"Overscan x/y: {os_x}/{os_y}")

        #convert to pygame surface and scale up raw tile render to interpolated zoom size
        start = time.perf_counter()
        pygame.surfarray.array_to_surface(buffer, render)
        debug.append(f"Arr -> surf: {(time.perf_counter() - start)*1000:.2f}")
        
        start = time.perf_counter()
        if self.zoom:
            scaled = pygame.transform.scale_by(buffer, (1+(self.zoom/self.zoom_steps))) 
            result.blit(scaled, np.divide(np.subtract(result.size, scaled.size), 2))
            debug.append(f"Scaled res: {scaled.size}")
        else:
            result.blit(buffer, np.divide(np.subtract(result.size, buffer.size), 2))
        #result.blit(pygame.transform.grayscale(buffer), np.divide((self.size[0] - buffer.get_width(), self.size[1] - buffer.get_height()), 2)) #debug
        debug.append(f"Buffer res: {buffer.size}")
        debug.append(f"Scale time: {(time.perf_counter() - start)*1000:.2f}")
        self.last_frame = result

        # ----------------------------------------------------------------------- tile streaming

        if self.debug_streaming:
            stream_debug_surf = pygame.Surface((self.pm_arr.shape[1], self.pm_arr.shape[0]))
            finished_streaming = np.zeros(max(self.unique_tiles)+1, np.uint8)
            for tile_idx in self.unique_tiles:
                for i, tile_store in enumerate(reversed(self.sm.tile_stores.values())):
                    i = len(self.sm.tile_stores) - (i+1)
                    if tile_idx in tile_store.finished:
                        finished_streaming[tile_idx] = i
                        break
            stream_debug = mosaic.render_funcs.debug_streaming(self.pm_arr, finished_streaming, self.max_size)
            pygame.surfarray.array_to_surface(stream_debug_surf, stream_debug)
            result.blit(stream_debug_surf, ui.util.graphics.centre(result.size, stream_debug_surf.size))

        start = time.perf_counter()
        if self.tile_size != self.min_size: 
            self.sm.stream(self.pm_arr_cropped, self.tile_size)
        debug.append(f"Pending processing time: {(time.perf_counter() - start)*1000:.2f}")
        debug.append(f"Min/used/total RAM budget: {int(self.sm.min_ram/(1024**2))}/{int(self.sm.ram_usage/(1024**2))}/{int(self.sm.init_ram/(1024**2))}")
        for line, value in self.sm.debug_strings.items():
            debug.append(f"{line}: {value}")
        if self.sm.prog_string:            
            result.blit(self.uii.fonts[16].render((f"Loading {self.sm.prog_counter}/{self.sm.prog_string}"), 1, [255]*3, [0]*3), (0,0))
        self.debug_strings = debug
        return result

    def debug_render(self):
        """to be used with a debugger
        \nwill raise AssertionError at the index of any tile that hasn't streamed in and exactly where it failed"""
        if self.sm.bg_thread.is_alive(): 
            return
        for tile_idx in np.unique(self.pm_arr_cropped):
            assert tile_idx in self.sm.tile_stores[self.tile_size] #streaming system has streamed in tile
            assert (np.any(self.sm.tile_stores[self.tile_size][PySaic.mosaic.tile_map[tile_idx]]) ==  #tile is valid 
                    np.any(PySaic.mosaic.tiles[tile_idx].as_fixed_res(self.tile_size)))    #(black or has otherwise been loaded in)

    def change_zoom(self, delta):
        new_size, new_zoom = divmod(int(math.log2(self.tile_size)*self.zoom_steps + self.zoom + delta), self.zoom_steps)
        new_size = 2**new_size
        
        if new_size < self.min_size: #zoom is out of bounds
            return int(math.log2(self.min_size) * self.zoom_steps)
        if new_size > self.max_size:
            return int(math.log2(self.max_size*2) * self.zoom_steps) - 1
        
        if new_size != self.tile_size:
            self.sm.halt()

        #refresh min_size tiles only once since streaming system skips it for every frame 
        if (new_size == self.min_size and new_size != self.tile_size and (PySaic.mosaic.mode != "vid" or not self.is_playing)): 
            self.sm.stream(self.unique_tiles, new_size, wait=True)

        self.tile_size = int(new_size)
        self.zoom = new_zoom
        self.last_frame = None #causes a redraw

    def change_pos(self, new):
        """make the mosaic move"""
        #pic
        if PySaic.mosaic.mode == "pic" or new == self.frame_pos: 
            return new #picture doesnt move
        #hyb
        if PySaic.mosaic.mode == "hyb": 
            self.frame_pos = new 
            self.last_frame = None #causes a redraw
            return new #hybrid tiles are all loaded in ahead of time 
        
        #vid - need to handle buffering and tile streaming
        if self.is_buffering and self.source.stream_buffer: #there are frames waiting to be processed
            self.source.check_frame(new) #make sure the thing hasn't crashed
            return self.frame_pos
        new = new % self.source.frame_count
        if new != self.frame_pos+1 or not self.source.processing_thread.is_alive(): #force a recovery
            self.source.seek(new)
            new_frame = self.source.read_frame(hold=True)
        elif isinstance(new_frame := self.source.read_frame(), bool): #frame isn't ready yet
            self.is_buffering = True
            return self.frame_pos
        #refresh min size tiles every 4 frames
        if self.tile_size == self.min_size and (not new%4 or not self.is_playing):
            self.sm.stream(self.unique_tiles, self.tile_size)
        self.last_frame = None #causes a redraw
        self.is_buffering = False
        self.pm_arr = new_frame #changes the match array being drawn by the renderer
        self.frame_pos = new
        return new
    
    def pan(self, delta):
        x, y = delta
        self.x_offset += x
        self.y_offset += y
        self.last_frame = None #causes a redraw