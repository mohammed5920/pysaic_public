from typing import Literal

import pygame
import numpy as np
import cv2

def centre(parent : tuple, child : tuple):
    return np.divide(np.subtract(parent, child), 2)
    
def blur_transparent(surface : pygame.Surface, k_size, padding, scale=1, brighten_by=0):
    bg = pygame.Surface(np.add(surface.size, (padding, padding)), pygame.SRCALPHA)
    bg.blit(surface, (padding/2,padding/2))
    if scale:
        bg = pygame.transform.smoothscale_by(bg, 1/scale)
    bg = pygame.transform.gaussian_blur(bg, k_size)
    
    #brightening pass
    if brighten_by:
        bg_array = pygame.surfarray.pixels3d(bg).astype(np.uint16)
        alpha = pygame.surfarray.array_alpha(bg).astype(np.uint16)
        bg_array = np.clip(bg_array * brighten_by, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha * brighten_by, 0, 255).astype(np.uint8)
        pygame.surfarray.array_to_surface(bg, bg_array)
        bg.convert_alpha()
        np.add(np.zeros_like(alpha), alpha, pygame.surfarray.pixels_alpha(bg))

    if scale > 1:
        bg = pygame.transform.smoothscale_by(bg, scale)
    return bg

def blur_quick(surface, k_size, scale=1):
    if not k_size % 2:
        k_size += 1
    scaled = pygame.transform.scale_by(surface, 1/scale)
    array = pygame.surfarray.pixels3d(scaled)
    edited = cv2.GaussianBlur(array, (k_size, k_size), 5)
    del array
    pygame.surfarray.blit_array(scaled, np.array(edited))
    return pygame.transform.scale_by(scaled, scale)

def scale_surface(surface, target_res, mode : Literal["fill", "fit"]):
    transparent = surface.get_alpha() is not None
    if mode == "fill":
        scale_factor = max(target_res[0] / surface.width, target_res[1] / surface.height)
    elif mode == "fit":
        scale_factor = min(target_res[0] / surface.width, target_res[1] / surface.height)

    scaled = pygame.transform.smoothscale_by(surface, scale_factor)
    if mode == "fit":
        return scaled
    
    if transparent:
        bg = pygame.Surface(target_res, pygame.SRCALPHA)
    else:
        bg = pygame.Surface(target_res)

    bg.blit(scaled, centre(target_res, scaled.size))
    return bg

def smart_colour(surface):
    consider_alpha = surface.get_flags() & pygame.SRCALPHA
    r,g,b = pygame.transform.average_color(surface, consider_alpha=consider_alpha)[:3]
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > 127:
        smart_colour = np.divide((r,g,b), 2)
    else:
        smart_colour = np.multiply((r,g,b), 2)
    return np.clip(smart_colour, 64, 192).astype(np.uint8)

def blend(a : pygame.Surface, b : pygame.Surface, strength):
    if a.size != b.size:
        b = pygame.transform.smoothscale(b, a.size)

    if strength <= 0:return a
    if strength >= 1:return b
    a, b = a.copy(), b.copy()
    b.set_alpha(255*strength)
    a.blit(b, (0,0))
    return a

def blend_greyscale(surface, strength):
    return blend(surface, pygame.transform.grayscale(surface), strength)

def coloured_square(colour, size, alpha=None):
    result = pygame.Surface(size, pygame.SRCALPHA if alpha is not None else 0)
    if alpha:
        colour = [*colour, alpha]
    result.fill(colour)
    return result