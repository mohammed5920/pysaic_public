import numba
import numpy as np
import os

#main functions

def draw(x_start : int, x_len : int, y_start : int, y_len : int, 
        tile_map : np.ndarray, tile_store : dict | np.ndarray, match_arr : np.ndarray, tile_size : int,
        is_hybrid : bool, frame_pos : int, end_frame_map : np.ndarray,
        st_flag : bool):
    """dispatch the correct render func to use according to y_len, is_hybrid, the type of tile_store, and st_flag \n
    input: \n
        x_start, y_start: which x/y index to start drawing the mosaic from
        x_len, y_len: how many tiles to draw on the x/y axis \n
        tile_map: map the index in match_list to the indices stored in tile_store (check stage_mosaic for more info)
        tile_store: array/dict where tiles are stored to be drawn from
        match_arr: array of tile indices representing the converted RGB colours of the original source
        tile_size: how big the tiles being drawn are \n
        is_hybrid: whether or not the tiles move
        frame_pos: no. of elapsed frames to offset the moving tiles at (not used if not is_hybrid)
        end_frame_map: map that stores the length of each moving tile (not used if not is_hybrid) \n
        st_flag: force the use of only a single thread\n
    output:\n
        a 3d array representing (part of) the whole mosaic
    """
    is_hybrid = int(is_hybrid)
    is_st = int(st_flag or y_len <= os.cpu_count()) #only use a single thread if mosaic is sufficiently small or if requested
    is_fast = int(isinstance(tile_store, np.ndarray)) #use a different renderer if tiles are stored smart (in a dict) or fast (in an array)
    mask = is_fast*100 + is_hybrid*10 + is_st #fast/hybrid/st
    match mask:
        case 0: 
            return draw_smart_static(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size)
        case 1:
            return draw_smart_static_st(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size)
        case 10:
            return draw_smart_hybrid(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size, frame_pos, end_frame_map)
        case 11:
            return draw_smart_hybrid_st(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size, frame_pos, end_frame_map)
        case 100:
            return draw_fast_static(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size)
        case 101:
            return draw_fast_static_st(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size)
        case 110:
            return draw_fast_hybrid(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size, frame_pos, end_frame_map)
        case 111:
            return draw_fast_hybrid_st(x_start, x_len, y_start, y_len, tile_map, tile_store, match_arr, tile_size, frame_pos, end_frame_map)

@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def debug_streaming(match_arr : np.ndarray, finished_streaming : np.ndarray, max_size : int):
    """inputs:
        match_arr: array of tile indices representing the converted RGB colours of the original source
        finished_streaming: array of len(max(match_arr)) where the index is a tile index and the value is the highest res tile_store it's contained in
        max_size: largest size the streaming system can store at, used to calculate the shades of white
    returns:
        greyscale 3d array where brighter pixels correspond to higher res tiles"""
    max_size = np.log2(max_size)
    mosaic = np.zeros((match_arr.shape[1], match_arr.shape[0], 3), dtype=np.uint8)
    for y in numba.prange(match_arr.shape[0]):
        for x in range(match_arr.shape[1]):
            match_id = match_arr[y, x]
            mosaic[x, y] = 255/max_size * finished_streaming[match_id]
    return mosaic

@numba.jit(nopython=True, nogil=True, cache=True)
def unique_nogil(array):
    """returns set(np.unique()) of the input ndarray\n
    does this in a numba accelerated (quicker set() iteration) function that releases the GIL"""
    return set(np.unique(array))

#------------------------------------ render functions ------------------------------------------------------
#splitting them this way looks silly but is 20/30% faster than adding the 3 if conditions inside the hot loop

@numba.jit(nopython=True, nogil=True, parallel=True, cache=True) #base function
def draw_fast_static(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_store : np.ndarray, match_list : np.ndarray, tile_size : int):

    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8) 

    #bounds checking because any error here leads to a silent crash that's a nightmare to debug
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):

            match_id = match_list[y, x] 
            mapped_id = tile_map[match_id] #map from tile_idx stored in match_list to mapped_id stored in tile_store (saves committed ram)
            mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                    (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile_store[mapped_id]

    return mosaic   

#differences in the rest of the functions:
#smart -> uses a dict instead of an array for tile_store (becomes tile_dict)
#st -> decorator is changed to omit parallel=True (function stays the same)
#hybrid -> mapped_id += frame_pos%end_frame_map[match_id] (makes the tile move while staying in bounds of its length)

@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def draw_smart_static(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_dict : dict, match_list : np.ndarray, tile_size : int): #<- difference here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id]
            tile = tile_dict.get(mapped_id) #<- and here
            if tile is not None: #<- and here
                mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                        (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile #<- and here
    return mosaic   
@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def draw_fast_hybrid(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_store : np.ndarray, match_list : np.ndarray, tile_size : int,
            frame_pos : int, end_frame_map : np.ndarray): #<- difference here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id] + frame_pos%end_frame_map[match_id] #<- and here
            mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                    (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile_store[mapped_id]
    return mosaic
@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def draw_smart_hybrid(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_dict : dict, match_list : np.ndarray, tile_size : int, #<- difference here
            frame_pos : int, end_frame_map : np.ndarray): #<- and here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id] + frame_pos%end_frame_map[match_id] #<- and here
            tile = tile_dict.get(mapped_id) #<- and here
            if tile is not None: #<- and here
                mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                        (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile #<- and here
    return mosaic

#single threaded versions (identical apart from decorators)

@numba.jit(nopython=True, nogil=True, cache=True) #<- difference here
def draw_fast_static_st(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_store : np.ndarray, match_list : np.ndarray, tile_size : int):
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id]
            mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                    (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile_store[mapped_id]
    return mosaic   
@numba.jit(nopython=True, nogil=True, cache=True) #<- difference here
def draw_smart_static_st(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_dict : dict, match_list : np.ndarray, tile_size : int): #<- and here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id]
            tile = tile_dict.get(mapped_id) #<- and here
            if tile is not None: #<- and here
                mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                        (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile #<- and here
    return mosaic   
@numba.jit(nopython=True, nogil=True, cache=True) #<- difference here
def draw_fast_hybrid_st(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_store : np.ndarray, match_list : np.ndarray, tile_size : int,
            frame_pos : int, end_frame_map : np.ndarray): #<- and here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id] + frame_pos%end_frame_map[match_id] #<- and here
            mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                    (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile_store[mapped_id]
    return mosaic
@numba.jit(nopython=True, nogil=True, cache=True) #<- difference here
def draw_smart_hybrid_st(x_start : int, x_len : int, y_start : int, y_len : int,
            tile_map : np.ndarray, tile_dict : dict, match_list : np.ndarray, tile_size : int, #<- and here
            frame_pos : int, end_frame_map : np.ndarray): #<- and here
    mosaic = np.zeros((x_len*tile_size, y_len*tile_size, 3), dtype=np.uint8)
    for y in numba.prange(max(y_start, 0), min(y_start+y_len, match_list.shape[0])):
        for x in range(max(x_start, 0), min(x_start+x_len, match_list.shape[1])):
            match_id = match_list[y, x]
            mapped_id = tile_map[match_id] + frame_pos%end_frame_map[match_id] #<- and here
            tile = tile_dict.get(mapped_id) #<- and here
            if tile is not None: #<- and here
                mosaic[(x-x_start) * tile_size:(x-x_start + 1) * tile_size,
                        (y-y_start) * tile_size:(y-y_start + 1) * tile_size] = tile #<- and here
    return mosaic