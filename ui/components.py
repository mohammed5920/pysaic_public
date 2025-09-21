from typing import Literal

from bidict import bidict
import pygame
import numpy as np

import ui.base
import ui.transform
import ui.util.exceptions
import ui.util.graphics
import ui.util.wrappers
import ui.util.mth

AlignType = Literal["topleft", "topmid", "topright", "midleft", "midmid", "midright", "botleft", "botmid", "botright"]

class PlainSurface(ui.base.Component):
    """pygame surfaces wrapped in a component, allows them to interact with the mouse and layout system"""
    def __repr__(self):
        return f"PlainSurface ({self.debug_colourkey})"
    
    def __init__(self, uii, surface : pygame.Surface=None, **kwargs):
        super().__init__(uii, **kwargs)
        self.surface = surface
    
    def get_surf(self, with_debug):
        result = self.surface
        return result
    
class Overlay(PlainSurface):
    def __repr__(self):
        return f"Overlay ({self.debug_colourkey})"
    
    def __init__(self, uii, **kwargs):
        if not uii.display:
            raise ui.util.exceptions.UIRenderingException("Can't use Overlay when UI instance is not initialised with reference to pygame display!")
        super().__init__(uii, None, **kwargs)
        self.screenshot = uii.display.copy()
        self.is_occluder = True
        self.surface = ui.util.graphics.blur_quick(self.screenshot, 31, scale=uii.settings.BLUR_DOWNSCALE)

    def resize(self, xy):
        screenshot = ui.util.graphics.scale_surface(self.screenshot, np.multiply(self.screenshot.size, xy), "fit")
        self.surface = ui.util.graphics.blur_quick(screenshot, 31, scale=self.uii.settings.BLUR_DOWNSCALE)

class BlurredSurface(PlainSurface):    
    def __repr__(self):
        return f"BlurredSurface ({self.debug_colourkey})"
    
    def __init__(self, uii, strength, brighten=0, 
                 surface=None, colour=None, size=None, 
                 padding=None, **kwargs):
        super().__init__(uii, surface, **kwargs)
        self.blur(strength, brighten, surface, colour, size, padding)
        
    def blur(self, strength, brighten=0, 
                 surface=None, colour=None, size=None, 
                 padding=None):
        if padding is None:
            padding = self.uii.scaler.min(self.uii.settings.BLOOM_PADDING)
        if not surface:
            surface = pygame.Surface(size)
            surface.fill(colour)
        self.surface = ui.util.graphics.blur_transparent(surface, strength, padding, self.uii.settings.BLUR_DOWNSCALE, brighten)

    def resize(self, xy):
        raise ui.util.exceptions.UIRenderingException("Call blur() on the new surface you want to blur instead.") 
        #quality gets bad if resizing blurred surfaces

class MediaSurface(PlainSurface):
    def __repr__(self):
        return f"MediaSurface ({self.wrapper.path})"
    
    def __init__(self, uii, media_path, scale_type : Literal["fit", "fill"], base_res = None, **kwargs):
        super().__init__(uii, **kwargs)
        self.wrapper = ui.util.wrappers.wrap_media(media_path, scale_type, self.uii.scaler.xy_min(base_res))
        self.res = base_res if base_res is not None else self.wrapper.size
        self.force_gc = True
        self.set_frame(0)

    def __len__(self):
        return len(self.wrapper)

    def __getitem__(self, idx):
        return self.wrapper[idx]

    def resize(self, xy):
        self.wrapper.resize(self.uii.scaler.xy_min(self.res))
        self.surface = self.wrapper[self.frame_idx]

    def set_frame(self, frame_idx):
        self.frame_idx = frame_idx%len(self.wrapper)
        self.surface = self.wrapper[frame_idx]

    def next_frame(self):
        self.frame_idx = (self.frame_idx+1)%len(self.wrapper)
        self.surface = self.wrapper[self.frame_idx]

class TextSurface(PlainSurface):
    def __repr__(self):
        return f"TextSurface ({self.text})"
    
    def __init__(self, uii, 
                 text, size, colour, 
                 bg=None, shadowed=False, **kwargs):
        """wraps pygame text api into something easier to work with here
        args: 
            uii, text, colour, size: self explanatory
            bg: colour of the text background"""
        super().__init__(uii, **kwargs)
        self.uii = uii
        self.text = text
        self.size = size
        self.colour = colour
        self.bg = bg
        self.shadowed = shadowed

        self.is_highlighted = False
        self._refresh()
        
    def _refresh(self):
        text = self.uii.fonts[self.size].render(self.text, True, self.colour, self.bg)

        if self.shadowed:
            black = self.uii.fonts[self.size].render(self.text, True, [0]*3, [0]*3 if self.bg is not None else None)
            shadowed = ui.util.graphics.blur_transparent(black, 4, 10)
            shadowed.blit(text, (5,5))
            self.surface = shadowed
        else:
            self.surface = text

        self.alt = self.uii.fonts[self.size].render(self.text, True, [0]*3, [255]*3)
        if self.is_highlighted: self.surface, self.alt = self.alt, self.surface

    def change_text(self, new_text):
        if new_text == self.text: return
        self.text = new_text
        self._refresh()

    def change_colour(self, new_colour):
        if ui.util.mth.comp_3d(self.colour, new_colour): return
        self.colour = new_colour
        self._refresh()

    def change_bg(self, new_bg):
        if ui.util.mth.comp_3d(self.bg, new_bg): return
        self.bg = new_bg
        self._refresh()

    def change_size(self, new_size):
        if new_size == self.size: return
        self.size = new_size
        self._refresh()

    def set_highlight(self, enable):
        if enable != self.is_highlighted:
            self.surface, self.alt = self.alt, self.surface
        self.is_highlighted = enable

    def resize(self, xy):
        self._refresh()

# -------------------------------------------------------------------------------------------------------

class LR_Layout(ui.base.Layout):
    def __repr__(self):
        return f"LR_Layout ({self.debug_colourkey})"
    
    """lays out components left to right"""
    def __init__(self, ui_instance, hz_padding=0, vr_padding=0, default_align="mid", **kwargs):
        super().__init__(ui_instance, **kwargs)
        self.default_align = default_align
        self.hz_padding = hz_padding
        self.vr_padding = vr_padding

        self.new_transform = ui.transform.DisappearHorz(self.uii.settings.FADE_TIME, reverse=True)
        self.del_transform = ui.transform.DisappearHorz(self.uii.settings.FADE_TIME)

    def get_surf(self, with_debug):
        if not self.components:
            return
        cum_width = max_height = 0
        surfs = []

        #pass 1
        for component in self.components.values():
            surf = component.render(with_debug)
            if surf is None: continue
            surfs.append(surf)
            max_height = max(max_height, surf.height)
            cum_width += surf.width
        if not cum_width: return

        #pass 2
        result = pygame.Surface((cum_width + self.uii.scaler.min(self.hz_padding)*(len(surfs)-1), max_height+self.uii.scaler.min(self.vr_padding)), pygame.SRCALPHA)
        prog_width = 0 #width progressively added to
        self.layout_map = {}
        for component, surf in zip(self.components.values(), surfs):
            tl_pos = np.array((prog_width, self.uii.scaler.min(self.vr_padding)/2))
            align = component.position.align[:3] if component.position else self.default_align
            tl_pos[1] += (max_height - surf.height) * 0.5 * ("top", "mid", "bot").index(align)
            prog_width += surf.width + self.uii.scaler.min(self.hz_padding)
            result.blit(surf, tl_pos)
            self.layout_map[component] = (tl_pos, np.add(tl_pos, surf.size))

        return result
    
class TB_Layout(ui.base.Layout):
    """lays out components top to bottom"""
    def __repr__(self):
        return f"TB_Layout ({self.debug_colourkey})"
    
    def __init__(self, ui_instance, hz_padding=0, vr_padding=0, default_align="mid", **kwargs):
        super().__init__(ui_instance, **kwargs)
        self.default_align = default_align
        self.hz_padding = hz_padding
        self.vr_padding = vr_padding

        self.new_transform = ui.transform.DisappearVert(self.uii.settings.FADE_TIME, reverse=True)
        self.del_transform = ui.transform.DisappearVert(self.uii.settings.FADE_TIME)

    def get_surf(self, with_debug):
        if not self.components:
            return None
        max_width = cum_height = 0
        surfs = []

        #pass 1
        for component in self.components.values():
            surf = component.render(with_debug)
            if surf is None: continue
            surfs.append(surf)
            max_width = max(max_width, surf.width)
            cum_height += surf.height
        if not cum_height: return

        #pass 2
        height = max(cum_height + self.uii.scaler.min(self.vr_padding)*(len(surfs)-1), 1)
        result = pygame.Surface((max_width+self.uii.scaler.min(self.hz_padding), height), pygame.SRCALPHA)
        prog_height = 0 #height progressively added to
        for component, surf in zip(self.components.values(), surfs):
            tl_pos = np.array((self.uii.scaler.min(self.hz_padding)/2, prog_height)) #top left coordinate of the surface
            align = component.position.align[3:] if component.position else self.default_align
            tl_pos[0] += (max_width - surf.width) * 0.5 * ("left", "mid", "right").index(align)
            prog_height += surf.height + self.uii.scaler.min(self.vr_padding)
            result.blit(surf, tl_pos)
            self.layout_map[component] = (tl_pos, np.add(tl_pos, surf.size))
        
        return result
    
class BaseButton(ui.base.Layout):
    def __init__(self, uii, toggled=True, **kwargs):
        super().__init__(uii, **kwargs)
        self.is_toggled=bool(toggled)
        self.new_transform = self.del_transform = None

    def toggle(self, toggle=None):
        self.is_toggled = bool(toggle) if toggle is not None else not self.is_toggled
        return self.is_toggled

class PaneButton(BaseButton):
    """background with text centered like a window pane"""  
    def __repr__(self):
        return f"PaneButton ({self.bg.wrapper.path})"

    def __init__(self, uii, 
                 bg_path, bg_size,
                 text, text_size, 
                 text_colour=None, toggled=False, **kwargs):
        super().__init__(uii, toggled, **kwargs)
        self.raw_size = bg_size
        self.bw_strength = 0
        self.bg = MediaSurface(uii, bg_path, "fill", self.raw_size, parent=self)
        self.blurred_bg = BlurredSurface(uii, 1, 0, self.bg[0], padding=0, parent=self)
        
        smart_colour = text_colour if text_colour else ui.util.graphics.smart_colour(self.bg[0])       
        self.components = bidict({
            "bloom" : BlurredSurface(uii, 9, 1.6, self.bg[0], parent=self),
            "mixed_bg" : PlainSurface(uii, self.blurred_bg.surface, parent=self),
            "text" : TextSurface(uii, text, text_size, smart_colour, shadowed=True, parent=self)
        })

        self.hovered_in_bounds = False
        
    def resize(self, xy):
        self.bg.resize(xy)
        self.components["text"].resize(xy)
        self.components["bloom"].blur(9, 1.6, self.bg[0])
        self.blurred_bg.blur(1, 0, self.bg[0], padding=0)

    def while_hovered(self, translated_mouse_coords):
        self.hovered_in_bounds = not self.components["bloom"].is_hovered
    def on_exit(self):
        self.hovered_in_bounds = False
    def on_down(self):
        if self.hovered_in_bounds:
            self.components["text"].set_highlight(True)
    def on_up(self):
        self.components["text"].set_highlight(False)
    def on_click(self):
        if self.hovered_in_bounds and self.click_func:
            self.click_func(*self.click_args)

    def get_surf(self, with_debug):
        #handle special effects when hovering over component
        if (self.hovered_in_bounds or self.is_toggled):
            self.bw_strength = max(0, self.bw_strength-1)
            self.bg.next_frame()

        elif not (self.hovered_in_bounds or self.is_toggled) and self.bw_strength < self.uii.settings.BUTTON_TIME:
            self.bw_strength += 1
        
        if self.bw_strength == self.uii.settings.BUTTON_TIME:
            self.bg.set_frame(0)

        self.debug_strings = [f"Frame IDX: {self.bg.frame_idx}",
                              f"Loadable / Total: {self.bg.wrapper.max_len}/{len(self.bg)}"]

        self.components["mixed_bg"].surface = ui.util.graphics.blend(self.bg.surface, self.blurred_bg.surface, self.bw_strength/self.uii.settings.BUTTON_TIME)
        self.components["bloom"].surface.set_alpha(255 * (1-(self.bw_strength/self.uii.settings.BUTTON_TIME)))

        coloured = super().get_surf(with_debug)
        return ui.util.graphics.blend_greyscale(coloured, self.bw_strength/self.uii.settings.BUTTON_TIME)
            
class TileButton(BaseButton):
    """text aligned right media aligned left solid colour background"""  
    def __repr__(self):
        return f"TileButton ({self.media.wrapper.path})"

    def __init__(self, uii, 
                 media_path, bg_size,
                 text, text_size, 
                 text_colour=None, toggled=False, **kwargs):
        super().__init__(uii, toggled, **kwargs)
        self.raw_size = bg_size
        self.media = MediaSurface(uii, media_path, "fit", np.subtract(self.raw_size, 30), parent=self)
        self.media_colour = np.divide(pygame.transform.average_color(self.media[0], consider_alpha=True), 4)
        self.bg = PlainSurface(self.uii, ui.util.graphics.coloured_square(self.media_colour, 
                                                                           self.uii.scaler.xy_min(self.raw_size)), parent=self)

        self.components = bidict({
            "bloom" : BlurredSurface(uii, 9, 1.6, self.bg.surface, parent=self),
            "bg" : self.bg,
            "media" : self.media.place("midleft", "midleft", (self.uii.settings.BLOOM_PADDING/2 + 15, 0)),
            "text" : TextSurface(uii, text, text_size, text_colour if text_colour is not None else [255]*3, parent=self)
                                .place("midright", "midright", (-(self.uii.settings.BLOOM_PADDING/2 + 15), 0))
        })
        
        self.bw_strength = 0
        self.hovered_in_bounds = False
        
    def resize(self, xy):
        self.bg.surface = ui.util.graphics.coloured_square(self.media_colour, self.uii.scaler.xy_min(self.raw_size))
        self.components["bloom"].blur(9, 1.6, self.bg.surface)
        self.components["text"].resize(xy)
        self.media.resize(xy)

    def while_hovered(self, translated_mouse_coords):
        self.hovered_in_bounds = not self.components["bloom"].is_hovered
    def on_exit(self):
        self.hovered_in_bounds = False
    def on_down(self):
        if self.hovered_in_bounds:
            self.components["text"].set_highlight(True)
    def on_up(self):
        self.components["text"].set_highlight(False)
    def on_click(self):
        if self.hovered_in_bounds and self.click_func:
            self.click_func(*self.click_args)

    def get_surf(self, with_debug):
        if (self.hovered_in_bounds or self.is_toggled):
            self.bw_strength = max(0, self.bw_strength-1)
            self.media.next_frame()
            
        elif not (self.hovered_in_bounds or self.is_toggled) and self.bw_strength < self.uii.settings.BUTTON_TIME:
            self.bw_strength += 1
            self.media.set_frame(0)

        self.components["bloom"].surface.set_alpha(255 * (1-(self.bw_strength/self.uii.settings.BUTTON_TIME)))
        self.debug_strings = [f"Frame IDX: {self.media.frame_idx}",
                              f"Loadable / Total: {self.media.wrapper.max_len}/{len(self.media)}"]

        coloured = super().get_surf(with_debug)
        return ui.util.graphics.blend_greyscale(coloured, self.bw_strength/self.uii.settings.BUTTON_TIME)

# ----------------------------------------------------------------------------------------

class Toast(BaseButton):
    def __repr__(self):
        return f"Toast ({self.text.text})"
    def __init__(self, uii, text, text_colour, padding=20, duration=None, 
                 **kwargs):
        
        super().__init__(uii, **kwargs)
        self.text = TextSurface(uii, text, 24, text_colour, parent=self)
        self.components = bidict({
            "bg" : PlainSurface(uii, ui.util.graphics.coloured_square(np.divide(text_colour, 2).astype(np.uint8), 
                                                                      np.add(self.text.surface.size, uii.scaler.min(padding)), 
                                                                      alpha=92), parent=self),
            "text" : self.text
        })
        self.padding = padding
        self.duration = duration * uii.settings.FPS if duration else uii.settings.TOAST_TIME
        self.elapsed = 0
        self.hover_counter = 0

    def resize(self, xy):
        self.text.resize(xy)
        self.components["bg"].surface = ui.util.graphics.coloured_square(self.text.colour, 
                                                                         np.add(self.text.surface.size, 
                                                                             self.uii.scaler.min(self.padding)), 
                                                                         alpha=92)

    def get_surf(self, with_debug):
        self.elapsed += 1
        if self.elapsed >= self.duration:
            self.delete()
        result = super().get_surf(with_debug)
        pygame.draw.aaline(result, self.text.colour, (0, result.size[1]-1), 
                         (result.size[0] * ((self.duration - self.elapsed) / (self.duration)), result.size[1]-1))
        if not self.is_hovered: self.hover_counter = max(0, self.hover_counter-1)
        result.set_alpha(255 - 128*(self.hover_counter/self.uii.settings.FADE_TIME))
        return result

    def on_click(self):
        self.delete()
    def while_hovered(self, pos):
        self.hover_counter = min(self.hover_counter+1, self.uii.settings.FADE_TIME)