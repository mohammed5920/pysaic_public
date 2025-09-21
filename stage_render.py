from pysaic import PySaic, Stage
import mosaic.render_classes
import pygame
import ui.core
import random

import ui.util
import ui.util.exceptions
import ui.util.wrappers

#user facing wrapper for the renderer, handles all controls
#(except for click and drag)

class Render(Stage):
    def start(self):
        self.ui = dict()
        self.ui["renderer"] = mosaic.render_classes.Renderer(PySaic.UI)
        self.stock_fps = PySaic.fps
        self.ui["renderer"].is_playing = PySaic.mosaic.mode != "pic"
        self.held_keys = set()
        
        PySaic.UI.add(self.ui)
        PySaic.change_fps(self.ui["renderer"].fps)

    def update(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button > 3: #scroll wheel
                zoom_delta = 4 if event.button == 4 else -4
                self.ui["renderer"].change_zoom(zoom_delta)

            if event.type == pygame.KEYDOWN: #handle key presses
                if event.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN):
                    self.held_keys.add(event.key) #wasd up down

                if event.key == pygame.K_SPACE: #space bar
                    self.ui["renderer"].is_playing = not self.ui["renderer"].is_playing
                    if self.ui["renderer"].is_playing:
                        PySaic.change_fps(self.ui["renderer"].fps)
                    else:
                        PySaic.change_fps(self.stock_fps)
                
                if event.key == pygame.K_r:
                    self.ui["renderer"].debug_render()
                
                if event.key == pygame.K_e:
                    self.ui["renderer"].debug_streaming = not self.ui["renderer"].debug_streaming
                
                if event.key == pygame.K_q:
                    self.ui["renderer"].change_pos(random.randrange(0, PySaic.mosaic.source.frame_count))

                if event.key == pygame.K_ESCAPE: #pause rendering if user wants to change modes/media
                    self.ui["renderer"].is_playing = False
                    PySaic.change_fps(self.stock_fps)
                    PySaic.transfer_stage("main")

                if not self.ui["renderer"].is_playing and event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                    frame_idx = 0 #youtube style frame by frame seeking with , .
                    if event.key == pygame.K_PERIOD:
                        frame_idx = self.ui["renderer"].frame_pos + 1
                    elif event.key == pygame.K_COMMA:    
                        frame_idx = self.ui["renderer"].frame_pos - 1
                    try:    
                        self.ui["renderer"].change_pos(frame_idx)
                    except ui.util.exceptions.UIMediaException:
                        PySaic.UI.toast("Error seeking in video!")
                        PySaic.switch_stage("main")
                        return
            
            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN):
                    self.held_keys.discard(event.key)

        if self.ui["renderer"].is_playing:
            try:
                self.ui["renderer"].change_pos(self.ui["renderer"].frame_pos + 1)
            except ui.util.exceptions.UIMediaException:
                PySaic.UI.toast("Error seeking in video!")
                PySaic.switch_stage("main")
                return
        
        delta = max(16 // self.ui["renderer"].tile_size, 1)
        jump = [0,0]
        jump[0] += delta*(pygame.K_d in self.held_keys) - delta*(pygame.K_a in self.held_keys)
        jump[1] += delta*(pygame.K_s in self.held_keys) - delta*(pygame.K_w in self.held_keys)
        if jump[0] or jump[1]:
            self.ui["renderer"].pan(jump)

        if pygame.K_DOWN in self.held_keys or pygame.K_UP in self.held_keys:
            zoom_delta = 1*(pygame.K_UP in self.held_keys) - 1*(pygame.K_DOWN in self.held_keys)
            self.ui["renderer"].change_zoom(zoom_delta)
            
    def pause(self): #stop streaming anything if user hits ESCAPE
        self.ui["renderer"].sm.halt()

    def resume(self): #restart streaming if user changes nothing after hitting ESCAPE and returning
        if self.ui["renderer"].tile_size == self.ui["renderer"].min_size:
            self.ui["renderer"].sm.stream(self.ui["renderer"].unique_tiles, self.ui["renderer"].min_size, wait=True)

    def cleanup(self): #user starts displaying another mosaic
        PySaic.change_fps(self.stock_fps)
        if "renderer" in self.ui:    
            self.ui["renderer"].sm.cleanup() #delete all streamed in tiles because python doesn't reliably do it
        for value in self.ui.values():
            value.delete()
        del self.ui