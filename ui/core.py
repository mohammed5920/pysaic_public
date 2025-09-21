import time
import random

import pygame
import numpy as np

import ui.base
import ui.components
import ui.util.wrappers

#mini ui structure
# -> simplest building block is a component
# -> all it does is render a pygame surface
# -> and exposes functionality that the parent component can call on when it detects that a mouse event has occured within the comp bounds

# --> components can be nested in a Layout
# --> Layout is also a component in and of itself since they can also be nested
# --> it's the Layout's job to figure out which of its components have the mouse's attention 
# --> Layouts build a map of the components during the render phase and test against it in the event handling phase, ensuring the hit detection works even with dynamic components

class Settings:
    def __init__(self, fps, **kwargs): #constants
        #resolution and frame rate must be set when object is created
        self.FPS = fps
        self.ANIM_SPEED = kwargs.get("anim_speed", 1)
        self.BASE_RES = kwargs.get("base_resolution", (1600, 900))
        #private settings to make *my* life a bit easier
        self.BLOOM_PADDING = 50
        self.BLUR_DOWNSCALE = 2
        self.change_fps(self.FPS)

    def change_fps(self, new):
        #anim_speed is the only exposed setting, everything else is tied to it
        self.FPS = new
        self.FADE_TIME = 0.2 * self.ANIM_SPEED * new
        self.BUTTON_TIME = 0.2 * self.ANIM_SPEED * new
        self.TOAST_TIME = 5 * self.ANIM_SPEED * new

class ScaleManager:
    def __init__(self, base_res, display_res):
        self.base_res = base_res
        self.display_res = display_res
    
    def x(self, x : int):
        """scales x by the difference between base res x and display res x"""
        return x * (self.display_res[0] / self.base_res[0])
    
    def y(self, y : int):
        """scales y by the difference between base res y and display res y"""
        return y * (self.display_res[1] / self.base_res[1])

    def xy(self, xy : tuple[int, int]):
        """scales xy by the difference between base res and display res"""
        x, y = xy
        return np.array((self.x(x), self.y(y)))

    def min(self, scalar : int):
        """scales input x by the smallest ratio of base res and display res"""
        #this keeps text from getting too tiny or too big if the window is reshaped to be really long or really short but only on one axis
        if scalar < 0:
            return max(self.xy((scalar, scalar)))
        return min(self.xy((scalar, scalar)))

    def xy_min(self, xy : tuple[int, int]):
        """scales input xy by the smallest ratio of base res and display res"""
        return np.multiply(xy, self.min(1))

class UIInstance:
    def __init__(self, display_res, fps, **kwargs):
        self.settings = Settings(fps, **kwargs)
        self.scaler = ScaleManager(self.settings.BASE_RES, display_res)
        self.fonts = ui.util.wrappers.FontWrapper(self)
        
        self.root = ui.base.RootLayout(self)
        self.toasts = ui.components.TB_Layout(self, vr_padding=10).place("topmid", "topmid", (0, 100))
        self.root.add_components({None : self.toasts})
        
        self.elapsed_frames = 0
        self.debug = False
        self.display = kwargs.get("display_surf", None) #directly draw to the display, enables more involved components but should be optional for maximum library modularity
        self.logger = kwargs.get("logger", None) #logger can optionally be set to enable debugging, but it isn't necessary

    def toggle_debug(self):
        self.debug = (self.debug+1)%3

    def get_unique_colour(self):
        duped = True
        used = {component.debug_colourkey for component in self.root.flatten_comp_tree()}
        while duped:
            colour = [random.randint(0, 96) for _ in range(3)]
            colour_key = colour[0] << 16 | colour[1] << 8 | colour[2]
            if colour_key not in used:
                return colour, colour_key
            
    def add(self, component_dict : dict[str, ui.base.Component]):
        self.root.add_components(component_dict)
    def toast(self, text, colour=None):
        if colour is None: colour = [255]*3
        toast = ui.components.Toast(self, text, colour)
        self.toasts.add_components({None : toast})
        return toast

    def resize(self, new_res):
        self.settings.display_res = new_res
        self.scaler.display_res = new_res
        
        start = time.perf_counter()
        self.fonts.resize()
        self.root.resize(self.scaler.xy((1,1)))
        if self.logger: 
            self.logger.log(f"UI resized in {(time.perf_counter() - start) * 1000:.02f} ms.")

    def handle_events(self, events):
        mouse_clicked = False
        mouse_unclicked = False
        mouse_pos = np.array(pygame.mouse.get_pos())

        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                self.toggle_debug()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button < 4:
                mouse_clicked = True
            #ignore if mouse is clicked and unclicked within the same frame (my mouse!!!)
            if event.type == pygame.MOUSEBUTTONUP and event.button < 4 and not mouse_clicked: 
                mouse_unclicked = True
        
        self.root.while_hovered(mouse_pos)
        self.root.handle_mouse(mouse_pos, mouse_clicked, mouse_unclicked)

    def draw(self, update_time=None):
        start_time = time.perf_counter()

        if not self.display:
            display = pygame.Surface(self.settings.display_res, pygame.SRCALPHA)
        else:
            display = self.display

        display.fill([0]*3)

        if self.debug == 1:
            debug_overlay = pygame.Surface(display.size, pygame.SRCALPHA)
            debug_overlay.set_alpha(64)
        if self.debug == 2:
            alpha_debug = pygame.Surface(display.size, pygame.SRCALPHA)

        self.root.render(display if self.debug != 2 else alpha_debug, self.debug == 1)

        if self.debug == 2:
            alpha = pygame.surfarray.array_alpha(alpha_debug)
            grayscale_array = np.stack((alpha, alpha, alpha), axis=-1)
            pygame.surfarray.blit_array(display, grayscale_array)

        self.elapsed_frames += 1        
        frame_time = (time.perf_counter() - start_time) * 1000
        frame_time = self.fonts[24].render(f"Frame time: {frame_time:02f}ms", 1, [255]*3 if frame_time < 3.34 else [255,0,0])
        display.blit(frame_time, (self.scaler.display_res[0] - frame_time.width, self.scaler.display_res[1] - frame_time.height*2))
        
        if update_time:
            update_time = self.fonts[24].render(f"Update time: {update_time*1000:02f}ms", 1, [255]*3 if update_time*1000 < 1 else [255,0,0])
            display.blit(update_time, np.subtract(self.scaler.display_res, update_time.size))
        
        if self.debug == 1:
            for i in range(1, 6):
                pygame.draw.line(debug_overlay, [255]*3, (0, self.scaler.display_res[1]/6*i), (self.scaler.display_res[0], self.scaler.display_res[1]/6*i))
                pygame.draw.line(debug_overlay, [255]*3, (self.scaler.display_res[0]/6*i, 0), (self.scaler.display_res[0]/6*i, self.scaler.display_res[1]))
            display.blit(debug_overlay, (0,0))
        
        if not self.display:
            return display
        
    def update(self, events, debug_time=None):
        self.handle_events(events)
        return self.draw(debug_time)