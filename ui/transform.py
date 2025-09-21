import pygame

import ui.util.graphics

class Transform:
    def __init__(self, duration, reverse=False):
        self.duration = duration #in frames
        self.anim_counter = 0
        self.is_finished = False
        self.is_reversed = reverse
        self.max_transparency = 255

    def prog(self):
        prog = self.anim_counter / self.duration
        prog = 3 * prog**2 - 2 * prog**3
        if self.is_reversed: return 1 - prog
        return prog
    
    def transform(self, surface):
        self.anim_counter += 1
        self.is_finished = self.anim_counter >= self.duration

    def copy(self, max_transparency):
        new = self.__class__.__new__(self.__class__) 
        new.__dict__ = self.__dict__.copy()
        new.max_transparency = max_transparency           
        return new
        
class FadeIn(Transform):
    def transform(self, surface):
        super().transform(surface)
        surface.set_alpha(self.max_transparency * self.prog())
        return surface

class DisappearVert(Transform):
    def transform(self, surface):
        super().transform(surface)
        height = surface.height*(1-self.prog())
        result = pygame.Surface((surface.width, height), pygame.SRCALPHA)
        result.set_alpha(self.max_transparency*(1-self.prog()))
        result.blit(surface, (0, -((surface.height-height)/2)))
        return result

class DisappearHorz(Transform):
    def transform(self, surface):
        super().transform(surface)
        width = surface.width*(1-self.prog())
        result = pygame.Surface((width, surface.height), pygame.SRCALPHA)
        result.set_alpha(self.max_transparency*(1-self.prog()))
        result.blit(surface, (-((surface.width-width)/2), 0))
        return result