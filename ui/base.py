from typing import Literal, TYPE_CHECKING
import gc

import pygame
import numpy as np
from bidict import bidict

import ui.transform
import ui.util.wrappers
import ui.util.exceptions
import ui.util.mth
if TYPE_CHECKING:
    from ui.core import UIInstance

AlignType = Literal["topleft", "topmid", "topright", "midleft", "midmid", "midright", "botleft", "botmid", "botright"]

class Position:
    def __init__(self, align : AlignType, anchor : AlignType, xy = None):
        """where does this component go?
        align: which point on the component are we talking about?
        anchor: where should that point "stick" to on the parent?
        xy: offset for that point"""
        self.align = align
        self.anchor = anchor
        self.offset = list(xy) if xy is not None else [0,0]

    def convert_align(self, surface_size, scale_xy):
        """returns the top left coordinate + offset of the surface"""
        offset = np.multiply(self.offset, scale_xy)
        adjusted = (surface_size[0] * 0.5 * ("left", "mid", "right").index(self.align[3:]), 
                    surface_size[1] * 0.5 * ("top", "mid", "bot").index(self.align[:3]))
        return np.subtract(offset, adjusted)
    
    def convert_anchor(self, surface_size, parent_size, scale_xy):
        """returns the top left coordinate of the surface offset relative to its parent"""
        top_left = self.convert_align(surface_size, scale_xy)
        adjusted = (parent_size[0] * 0.5 * ("left", "mid", "right").index(self.anchor[3:]), 
                    parent_size[1] * 0.5 * ("top", "mid", "bot").index(self.anchor[:3]))
        return np.add(top_left, adjusted)

class Component:
    """any class inheriting from Component must override
    resize() and render()"""
    def __eq__(self, value): #for when calling UIInstance.components.remove(component)
        return value.debug_colourkey == self.debug_colourkey
    def __hash__(self):
        return self.debug_colourkey

    def __init__(self, ui_instance, **kwargs):
        #used to draw
        self.uii : 'UIInstance' = ui_instance
        self.parent : Layout = kwargs.get("parent", None)
        self.position : Position = kwargs.get("position", Position("midmid", "midmid"))
        self.transforms : list[ui.transform.Transform] = []

        #set once
        self.click_func = kwargs.get("click_func", None) #function executed when component is clicked
        self.click_args = kwargs.get("click_args", []) #args passed to click_func when clicked
        self.hover_func = kwargs.get("hover_func", None) #same as above but for hovering
        self.hover_args = kwargs.get("hover_args", []) #same as above but for hovering
        self.force_gc = kwargs.get("force_gc", False)
        self.max_transparency = kwargs.get("alpha", 255)
        self.is_layout = False
        
        #used to track state externally
        self.is_clicked = False
        self.is_hovered = False
        self.is_alive = True

        self.debug_strings = []

        #hack for root layout
        #since getting colour depends on flattening root tree
        #but root is not fully initialised when getting colour
        if not kwargs.get("no_colour", False):
            self.debug_colour, self.debug_colourkey = ui_instance.get_unique_colour()
        else:
            self.debug_colour = [0]*3
            self.debug_colourkey = 0

    # ------ positioning and scaling 

    def place(self, align : AlignType, anchor : AlignType, xy = None):
        """where does this component go?
        align: which point on the component are we talking about?
        anchor: where should that point "stick" to on the parent?
        xy: offset for that point"""
        self.position = Position(align, anchor, xy)
        return self

    def resize(self, xy):
        """called when the window is resized
        args:
            xy tuple: scale of the new res width/height compared to the base"""
        raise NotImplementedError

    # ------ manage kids/parents 

    def get_recursive_property(self, property):
        if self.__dict__[property]: return True
        if not self.is_layout: return False
        for component in self.components.values():
            if component.get_recursive_property(property): return True
        return False

    def cleanup(self):
        """runs once the component is deleted"""
        pass

    def delete(self):
        """queue the component for deletion
        this defers deletion to the parent's deletion handler"""
        self.parent.queue_del_component(value=self)

    # ------ manage mouse

    def on_exit(self):
        """executed when the mouse exits the component boundary"""
        pass

    def on_down(self):
        """executed when the mouse is clicked (mouse down) within the component boundary
        on_click() is preferred since this is what emulates what other libraries do"""
        pass

    def on_up(self):
        """executed when the mouse is unclicked (mouse up) when the component was the last to be clicked, 
        regardless of whether or not the mouse is still in the component boundary
        on_click() is preferred since this is what emulates what other libraries do"""
        pass

    def on_click(self):
        """executed when the mouse is clicked and then unclicked WITHIN the component boundary"""
        if self.click_func:
            self.click_func(*self.click_args)

    def on_enter(self):
        """executed when the mouse first enters the component boundary"""
        if self.hover_func:
            self.hover_func(*self.hover_args)
    
    def while_hovered(self, translated_mouse_coords):
        """runs every frame the mouse is in the component boundary"""
        pass

    def while_clicked(self, translated_mouse_coords):
        """run every frame the component is clicked"""
        pass
    
    # ------ drawing / updating state

    def get_surf(self, with_debug):
        """return the raw pygame surface of the component, runs every frame the component is visible for"""
        raise NotImplementedError

    def debug(self, rendered_surf):
        """render debug info overlay to the rendered surface"""
        debug_surf = pygame.Surface(rendered_surf.size, pygame.SRCALPHA)
        debug_surf.fill(self.debug_colour)
        debug_surf.set_alpha(128)
        rendered_surf.blit(debug_surf, (0,0))

        t_height = 0
        for line in self.debug_strings:
            d_text = self.uii.fonts[16].render(line, 1, [255]*3)
            rendered_surf.blit(d_text, (0, t_height))
            t_height += d_text.height
        return rendered_surf

    def render(self, with_debug):
        """returns processed pygame surface of the component, runs every frame the component is visible for"""
        raw_surf = self.get_surf(with_debug)
        if not raw_surf: return 

        if with_debug or self.transforms:
            render_copy = raw_surf.copy()
            if with_debug: 
                render_copy = self.debug(render_copy)
            for i, transform in enumerate(self.transforms.copy()):
                render_copy = transform.transform(render_copy)
                if transform.is_finished:
                    self.transforms.remove(transform)
            return render_copy
        else:
            if self.max_transparency != 255:
                raw_surf.set_alpha(self.max_transparency)
            return raw_surf

class Layout(Component):
    """by default, layout all components relative to the first one
    \nif an inheritant wishes to render differently, it needs to also fill out self.layout_map - component reference : ( (top_left_x, y), (bottom_right_x, y) )
    \nalternatively, override handle_mouse() as well"""
    def __getitem__(self, key):
        return self.components[key]
    def __setitem__(self, key, value):
        self.add_components({key : value})
    def __contains__(self, key : str):
        return key in self.components and key not in self.decomponents

    def __init__(self, ui_instance, **kwargs):
        super().__init__(ui_instance, **kwargs)
        self.components : bidict[str, Component] = bidict()
        self.decomponents : dict[str, Component] = {} #for deferred deletion
        self.layout_map : dict[Component, tuple[tuple[int, int], tuple[int, int]]]= {} #component reference : ( (top_left_x, y), (bottom_right_x, y) )

        self.is_layout = True

        self.new_transform = ui.transform.FadeIn(self.uii.settings.FADE_TIME)
        self.del_transform = ui.transform.FadeIn(self.uii.settings.FADE_TIME, reverse=True)

    # ------ positioning / scaling

    def resize(self, xy):
        for component in self.components.values():
            component.resize(xy)

    # ------ manage kids/parents 

    def add_components(self, components : dict[str, Component], new_transform : ui.transform.Transform = None):
        """add all components in the dict to the layout, applying new_transform to them to fade them in (none for default, False to not add)"""
        for key, component in components.items():
            if key is None: key = str(component.debug_colourkey)
            if component.parent: raise ui.util.exceptions.UILayoutException(f"{component} can't have 2 parents!")
            component.parent = self
            
            if new_transform:
                component.transforms.append(new_transform.copy(component.max_transparency))
            elif self.new_transform:
                component.transforms.append(self.new_transform.copy(component.max_transparency))

            self.decomponents.pop(key, None) #adding a component with the same key as one that is being faded out
            self.components[key] = component
        return self

    def queue_del_component(self, key=None, value=None, del_transform : ui.transform.Transform = None):
        """schedule a component for deletion
        \n this will apply any animations (fadeouts,etc) and add it to be deleted later after the component has finished rendering (none for default, False to not add)"""
        if key is None and value is None:
            raise ui.util.exceptions.UIException("What do you want me to delete?")
        if key is None:
            key = self.components.inverse[value]
        if key in self.decomponents: return
        if value is None:
            value = self.components[key]
        
        value.is_alive = False
        if del_transform:
            value.transforms.append(del_transform.copy(value.max_transparency))
        elif self.del_transform:
            value.transforms.append(self.del_transform.copy(value.max_transparency))
        
        self.decomponents[key] = value

    def cleanup(self):
        for component in self.components.values():
            component.cleanup()
        del self.components

    def find_comp_under_point(self, point, level=0):
        """returns a reference to the component that point collides with"""
        for comp, bounds in reversed(self.layout_map.items()):
            tl, br = bounds
            if ui.util.mth.check_point_in_bounds(tl, point, br):
                if comp.is_layout:
                    return comp.find_comp_under_point(np.subtract(point, tl), level+1)
                else:
                    return comp, level
        if level:
            return self, level
        else: 
            return None, None
        
    def flatten_comp_tree(self, comps=None):
        """returns a list of all components that are in the layout, including ones in any child layouts"""
        if not comps: comps = [self]
        for comp in self.components.values():
            comps.append(comp)
            if comp.is_layout:
                comp.flatten_comp_tree(comps)
        return comps

    # ------ manage mouse 

    def handle_mouse(self, mouse_pos, clicked, unclicked):
        #whenever the mouse is in a component's area, while_hovered needs to be called. if its the first time entering, on_enter() gets called. 
        #if it was being hovered over but left on that exact frame, on_exit gets called

        #however, clicking is different. if a mouse is clicked within the bounds, is_clicked is set to true. if the mouse sends a MOUSEUP event, 
        #within the bounds, it causes on_click to happen. if the mouse is "unclicked" but out of bounds , it silently disables is_clicked. 
        #while is_clicked, while_clicked should always be running.
        first_hit = False

        for component, (tl, br) in reversed(self.layout_map.items()):
            if not component.is_alive: continue
            comp_hit = False
            translated = np.subtract(mouse_pos, tl)
            
            if component.is_clicked:
                component.while_clicked(translated)
                if unclicked:
                    component.on_up()

            if ui.util.mth.check_point_in_bounds(tl, mouse_pos, br) and not first_hit:
                comp_hit = True
                first_hit = True
                if not component.is_hovered: component.on_enter()
                component.is_hovered = True
                component.while_hovered(translated)
            
                #clicked inside boundary
                if clicked:
                    component.is_clicked = True #part of 2 stage click, wait for click & unclick to happen both within boundary
                    component.on_down()
                #unclicked inside boundary
                if component.is_clicked and unclicked:
                    component.is_clicked = False
                    component.on_click()
        
            else:
                if unclicked: #mouse unclicked somewhere else, do not go through with 2 stage click
                    component.is_clicked = False
                if component.is_hovered:
                    component.on_exit()
                component.is_hovered = False

            if component.is_layout:
                component.handle_mouse(translated if comp_hit else (-1, -1), clicked and comp_hit, unclicked)
        
        self.layout_map = {}

    # ------ manage rendering 

    def debug(self, rendered_surf):
        rendered_surf = super().debug(rendered_surf)
        for comp, (tl, br) in self.layout_map.items():
            pygame.draw.aacircle(rendered_surf, comp.debug_colour, tl, self.uii.scaler.min(3))
            pygame.draw.aacircle(rendered_surf, comp.debug_colour, br, self.uii.scaler.min(3))
        return rendered_surf

    def get_surf(self, with_debug):
        """by default, layout components in the layout relative to the first one stored in Layout.components
        \nlayouts need to handle if children return None for get_surf()"""
        if not self.components:
            return #tell parent to ignore the draw because layout has no components
        
        comps = list(self.components.values())
        surfs = [comp.render(with_debug) for comp in comps]
        if surfs[0] is None:
            raise ui.util.exceptions.UILayoutException("Cannot ignore rendering the first component in a layout!")
        
        result = pygame.Surface(surfs[0].size, pygame.SRCALPHA)
        self.layout_map[comps[0]] = ((0,0), result.size)
        result.blit(surfs[0], (0,0))

        for comp, surf in zip(comps[1:], surfs[1:]):
            if not comp.position:
                raise ui.util.exceptions.UILayoutException("Extra components in a layout need to have position info!")
            
            #ignore drawing this component because layout has no components
            if surf is None:
                continue
            
            tl_pos = comp.position.convert_anchor(surf.size, result.size, self.uii.scaler.xy_min((1,1)))
            result.blit(surf, tl_pos)
            self.layout_map[comp] = (tl_pos, np.add(tl_pos, surf.size))

        return result
    
    def render(self, with_debug):
        #override this here to add one extra bit of functionality
        #after rendering all comps in a layout, handle comps that were deleted via queue_delete
        result = super().render(with_debug)
        deleted = []
        force_gc = False
        for key, comp in self.decomponents.items():
            if not comp.transforms:
                force_gc = force_gc or comp.get_recursive_property("force_gc")
                comp.cleanup()
                self.components.pop(key)
                deleted.append(key)
        [self.decomponents.pop(key) for key in deleted]
        if force_gc:
            gc.collect()
        return result

class RootLayout(Layout):
    #special root layout component
    #the difference between this and normal layout is that this one treats the display surface as the "first component"
    #includes some hacks to get toasts working
    def __repr__(self):
        return "Root"

    def __init__(self, ui_instance, **kwargs):
        kwargs["no_colour"] = True
        super().__init__(ui_instance, **kwargs)
        self.debug_mouse_pos = None

    def while_hovered(self, translated_mouse_coords):
        self.debug_mouse_pos = translated_mouse_coords

    def render(self, display_surf, with_debug):
        comps = list(self.components.values())

        for comp in (*comps[1:], comps[0]):
            if not comp.position:
                raise ui.util.exceptions.UILayoutException("Root level components need to have position info!")
            
            surf = comp.render(with_debug)
            if surf is None: continue

            tl_pos = comp.position.convert_anchor(surf.size, self.uii.scaler.display_res, self.uii.scaler.xy_min((1,1)))
            display_surf.blit(surf, tl_pos)
            self.layout_map[comp] = (tl_pos, np.add(tl_pos, surf.size))

        if with_debug and self.debug_mouse_pos is not None:
            pygame.draw.aacircle(display_surf, [255]*3, self.debug_mouse_pos, 10)
            comp, level = self.find_comp_under_point(self.debug_mouse_pos)
            debug_search = self.uii.fonts[24].render(f"{level} | {comp} | {comp.parent if comp else None}", 1, [255]*3)
            display_surf.blit(debug_search, self.debug_mouse_pos)

        deleted = []
        force_gc = False
        for key, comp in self.decomponents.items():
            if not comp.transforms:
                force_gc = force_gc or comp.get_recursive_property("force_gc")
                comp.cleanup()
                self.components.pop(key)
                deleted.append(key)
        [self.decomponents.pop(key) for key in deleted]
        if force_gc:
            gc.collect()