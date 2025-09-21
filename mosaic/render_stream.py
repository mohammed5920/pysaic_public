from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import math
import time
import os

from PIL import Image
from numba import types
from numba import typed
from tqdm import tqdm
import numpy as np
import psutil
import cv2

from pysaic import PySaic
import mosaic.render_funcs
import mosaic.tiles
import mosaic.tiles_funcs
import util.misc

STAGES = ["Chilling", "Safe", "Troubling", "Critical", "lmao"]

class TileStore:
    """data structure that stores the tiles streamed in, as well as metadata for all the tiles
    \TileStores can be fast or smart, fast TileStores wrap a numpy array since accessing them is much quicker,
    but requires allocating space for every possible tile that can be loaded in
    \nfor large resolutions, or lots of tiles, smart mode wraps a dict that can shrink to store tiles only on screen
    """
    def __init__(self, tile_size : int, len : int, smart=False):
        """initliase a TileStore
        args:
            tile_size: how big the tiles are (only one size per TileStore)
            len: combined number of tiles and how many frames are in those tiles
            smart: toggle between wrapping a numpy array and wrapping a numba dict"""
        self.finished = OrderedDict() #tile_idx : (mapped idx, tile length)
        self.lock = threading.Lock() #numba dicts and OrderedDicts are not threadsafe so locking must be used
        self.tile_size = tile_size
        self.len = len
        self.is_smart = smart
        if not smart: #numpy arrays allocate all memory needed upfront, meaning fixed ram usage
            self.raw = np.zeros((len, tile_size, tile_size, 3), dtype=np.uint8)
            self.ram_usage = len*tile_size*tile_size*3
            util.misc.log(f"Size {tile_size}: Fast")
        else: 
            self.raw = typed.Dict.empty(key_type=types.int32, value_type=types.Array(types.uint8, 3, "C"))
            self.ram_usage = 0
            util.misc.log(f"Size {tile_size}: Smart")

    def add_tile(self, tile_idx : int, mapped_idx : int, values : list[np.ndarray]):
        """add a tile to the TileStore
        args:
            tile_idx: reference to tile in the global PySaic.mosaic.tiles list
            mapped_idx: where to map the tile into the array
                this is done to save on committed RAM, (check stage_mosaic for more info)
            values: a list of frames to add for that given tile (e.g frames of a single video tile)"""
        with self.lock:
            if tile_idx in self.finished:
                util.misc.log(f"Warning: Adding {tile_idx} in TileStore {self.tile_size} more than once...")
                return
            self.finished[tile_idx] = (mapped_idx, len(values))
            self.finished.move_to_end(tile_idx)
            for i, frame in enumerate(values):
                if self.is_smart: self.ram_usage += self.tile_size*self.tile_size*3
                self.raw[mapped_idx+i] = np.array(frame, order="C")

    def refresh(self, idxs : set[int]):
        """move all the tiles in the input set to very end of the LRU cache so they're evicted last"""
        if not self.is_smart: return #deleting items from an array doesn't save RAM
        with self.lock:
            for idx in idxs: 
                if idx not in self.finished: continue
                self.finished.move_to_end(idx)

    def get_least_used(self):
        """get the list of tile indices stored in the TileStore sorted by least used"""
        return list(self.finished.keys())

    def free_tile(self, tile_idx : int):
        """free a tile from the underlying data structure
        returns the amount of RAM freed by deleting the tile"""
        if not self.is_smart: return 0  #deleting items from an array doesn't save RAM
        freed = 0 
        with self.lock:
            if tile_idx not in self.finished: return 0
            mapped_idx, tile_len = self.finished.pop(tile_idx)
            for offset in range(mapped_idx, mapped_idx+tile_len):
                self.raw.pop(offset)
                freed += self.tile_size*self.tile_size*3
            self.ram_usage -= freed
        return freed

    def __getitem__(self, idx):
        return self.raw[idx]
    
    def __contains__(self, idx):
        return idx in self.finished
    
    def __enter__(self): #context manager is used only to read from the array, insuring no writes happening while being read from
        if self.is_smart: self.lock.acquire() 
        return self
    
    def __exit__(self, a,b,c):
        if self.is_smart: self.lock.release()

class StreamingManager:
    def __init__(self):
        
        """initialise the streaming manager"""
        self.iterating = False
        self.tile_stores : dict[int, TileStore] = {}
        self.fast_limit : int = 1 #largest size that uses a TileStore that is fast to read from
        self.init_ram = psutil.virtual_memory().available*(PySaic.settings.RAM_PERCENT/100) #75% (default) of free ram at init time

        #read only references shortened for Brevity
        self.tiles = PySaic.mosaic.tiles
        self.max_tile_len = PySaic.mosaic.max_tile_len
        self.tile_map = PySaic.mosaic.tile_map

        self.bg_thread : util.misc.ThreadWrapper = None
        self.stop_streaming_flag = False
        self.never_stream_flag = False
        
        #hacked together debug reporting
        self.debug_strings = {}
        self.prog_counter = 0
        self.prog_string = ""

    def partition(self, min_limit, display_res):
        self.res_limit : int = 2 ** math.floor(math.log2(min(display_res))) #max size that can be entirely displayed on screen

        total_frames = PySaic.mosaic.total_frames
        available_ram = self.init_ram
        
        #min_ram calculation is more of a guess
        #technically the MINIMUM needed to load is just the screen resolution
        #but we need to take into account that the streaming system loads in video tiles at 4x the requested resolution (upper bounded to res_limit ofc)
        #otherwise zooming in becomes a real chore
        #also
        #we need to split the pool between fixed and dynamic tile allocation
        #making the limit too generous means it can sometimes allocate the entire RAM pool as fixed leaving very little for high res dynamic allocations
        #also
        #for hybrid mode, the tiles move, and for dynamic allocations we need to consider that we need to allocate for those extra frames
        #which raises the question, how many frames do we prepare for? avg, or max?
        #streaming system cannot dynamically stream in vid tile frames as the hybrid renderer plays, only the entire tile
        #because otherwise CPU would always be pegged at 100% (video is Expensive)
        #so  
        #i feel like the best compromise is the size of 2 arrays at current size + tile dictionary at 4x(default) the screen resolution * avg no. of frames per tile

        self.min_ram = 2*total_frames*3 + 3*PySaic.settings.PREFETCH_MULTIPLIER*min(PySaic.mosaic.mean_tile_len, PySaic.mosaic.max_tile_len)*display_res[0]*display_res[1]*3
        if available_ram < self.min_ram: print(f"Might not be enough RAM - {self.min_ram//1024//1024} megabytes recommended...")

        self.ram_usage = 0
        kept = set()

        for exponent in range(int(math.log2(min_limit)), int(math.log2(self.res_limit*2))):
            raised = 2**exponent
            required = 2*total_frames*raised*raised*3 + 3*PySaic.settings.PREFETCH_MULTIPLIER*min(PySaic.mosaic.mean_tile_len, PySaic.mosaic.max_tile_len)*display_res[0]*display_res[1]*3
            kept.add(raised)
            if available_ram - required >= 0 or raised == 1:
                if raised not in self.tile_stores or self.tile_stores[raised].is_smart:
                    if raised in self.tile_stores: 
                        self.tile_stores[raised].raw = None
                    self.tile_stores[raised] = TileStore(raised, total_frames, smart=False)
                available_ram -= raised*raised*3*total_frames
                self.ram_usage += raised*raised*3*total_frames
                self.fast_limit = raised
            else:
                if raised not in self.tile_stores or not self.tile_stores[raised].is_smart:
                    if raised in self.tile_stores: 
                        self.tile_stores[raised].raw = None
                    self.tile_stores[raised] = TileStore(raised, total_frames, smart=True)

        for key in self.tile_stores.copy():
            if key not in kept:
                util.misc.log(f"Cleaning up TS size {key}...")
                ts = self.tile_stores.pop(key)
                ts.raw = None

    def get_store(self, size):
        return self.tile_stores[size]

    def halt(self):
        self.stop_streaming_flag = True
        if self.bg_thread: self.bg_thread.join()
        self.stop_streaming_flag = False

    def cleanup(self):
        self.halt()
        self.never_stream_flag = True
        for tile_store in self.tile_stores.values():
            tile_store.raw = None #forcefully dereference the raw array/dict so python's GC always cleans it up later (otherwise it sometimes wouldn't)

    def stream(self, tiles : np.ndarray | set, size : int, wait=False, stall=False):
        """start a background thread to stream in tiles
        
        Args:
        tiles: set or numpy array of indices that need to be streamed (duplicates are filtered out automatically)
        size: size of tiles to be streamed in
        wait: blocks calling thread until any previous streaming jobs are finished
        stall: blocks calling thread until THIS streaming job is finished
        """
        if wait and self.bg_thread: self.bg_thread.join() 
        if not self.bg_thread or not self.bg_thread.is_alive():
            self.bg_thread = util.misc.ThreadWrapper(target=self._stream_thread, args=(tiles, size))
            self.bg_thread.daemon = True
            self.bg_thread.start()
            if stall:
                self.bg_thread.join()

    def _find_supersized_tile(self, tile_idx : int, start_size : int): 
        """returns any sizes > start_size where higher res tile_idx has been loaded in, or False"""
        if start_size >= self.res_limit and tile_idx in self.tile_stores[self.res_limit]:
            return self.res_limit
        #special case - skip to starting probing from self.fast_limit immediately if start_size is under
        if start_size < self.fast_limit and tile_idx in self.tile_stores[self.fast_limit]:
            return self.fast_limit
        #otherwise, start at size*2 and probe until res_limit
        size = max(self.fast_limit, start_size)*2
        while size <= self.res_limit:
            if tile_idx in self.tile_stores[size]:
                return size
            size *= 2
        return False

    def _base_tile(self, tile_idx, ts):
        """gets tile_idx at ts resolution
        loaded in at matching time and stored in the tile object itself"""
        if self.stop_streaming_flag: return
        tile = self.tiles[tile_idx]
        mapped = self.tile_map[tile_idx]
        self.tile_stores[ts].add_tile(tile_idx, mapped, (tile.as_fixed_res(ts),))
        self.prog_counter += 1
        
    def _pic_tile(self, tile_idx, ts):
        """gets tile_idx at ts resolution
        from a picture file"""
        if self.stop_streaming_flag: return
        tile = self.tiles[tile_idx]
        mapped = self.tile_map[tile_idx]
                                    #open image, convert to rgb, turn into array, crop to square, resize to required size, invert horizontally and rotate 90 degrees
        self.tile_stores[ts].add_tile(tile_idx, mapped, 
                                    (np.rot90(np.fliplr(cv2.resize(mosaic.tiles_funcs.crop(np.array(Image.open(tile.path).convert("RGB")))
                                                                  , (ts, ts), interpolation=cv2.INTER_AREA))),))
        self.prog_counter += 1

    def _vid_tiles(self, vid_path, tile_idxs, ts):
        """gets tile_idxs at ts resolution
        from vid_path"""
        if self.stop_streaming_flag: return
        vid_cap = cv2.VideoCapture(vid_path)
        vid_size = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        dscale_factor = max(math.floor(np.min(np.divide(vid_size, ts))),1)
        tile_idxs = sorted(tile_idxs, key=lambda tile_idx: self.tiles[tile_idx].frame_idx)

        for tile_idx in tqdm(tile_idxs, desc=f"{vid_path} - streaming...", leave=False, mininterval=1):
            if self.stop_streaming_flag: return
            tile = self.tiles[tile_idx]
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, tile.frame_idx)
            mapped = self.tile_map[tile_idx]

            result = []
            for _ in range(mapped, mapped+min(tile.len, self.max_tile_len)):
                if self.stop_streaming_flag: return
                ret, extracted_frame = vid_cap.read()
                if not ret: 
                    util.misc.log(f"Error reading {vid_path}!")
                    break

                cropped_frame = cv2.resize(mosaic.tiles_funcs.crop(extracted_frame[::dscale_factor, ::dscale_factor, ::-1]), (ts, ts), 
                                            interpolation=cv2.INTER_LINEAR)
                result.append(np.rot90(np.fliplr(cropped_frame)))
            
            self.tile_stores[ts].add_tile(tile_idx, mapped, result)
            self.prog_counter += 1
        vid_cap.release()
    
    def _cached_tile(self, tile_idx, ts):
        """gets tile_idx at ts resolution
        from a higher resolution version of that tile"""
        if self.stop_streaming_flag: return
        cached_size = self._find_supersized_tile(tile_idx, ts)
        mapped = self.tile_map[tile_idx]
        tile = self.tiles[tile_idx]
        result = []
        for i in range(mapped, mapped+min(tile.len, self.max_tile_len)):
            big_tile = self.tile_stores[cached_size][i]
            if ts != 1:
                result.append(cv2.resize(big_tile, (ts, ts), interpolation=cv2.INTER_AREA))
            else:
                result.append(np.mean(big_tile, axis=(0, 1)).astype(np.uint8))
        self.tile_stores[ts].add_tile(tile_idx, mapped, result)
        self.prog_counter += 1

    def _dispatch(self, tile_set, tile_func, tile_size, tile_description):
        if self.stop_streaming_flag or not tile_set: 
            self.prog_string = ""
            return not tile_set
        self.prog_counter = 0
        self.prog_string = f"{len(tile_set)} {tile_description}..."
        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda idx: tile_func(idx, tile_size), tile_set)) #calling list() to execute it immediately ?
        self.prog_string = ""
        return True

    def _stream_thread(self, onscreen, ts):
        #preprocessing
        if self.stop_streaming_flag or self.never_stream_flag: return
        start = time.perf_counter()
        self.iterating = True
        if isinstance(onscreen, np.ndarray):
            onscreen = mosaic.render_funcs.unique_nogil(onscreen)
        self.iterating = False
        blanks = onscreen.difference(self.tile_stores[ts].finished.keys())
        self.debug_strings["Filtering time (threaded)"] = round((time.perf_counter() - start)*1000,2)

        self.prog_counter = 0 #used to track how many tiles / total have been loaded
        self.prog_string = "" #used to display on the screen which type of tile we're waiting on
        if not blanks: #no tiles need to be streamed, but we still need to update LRU cache statistics
            for tile_store in [tst for tst in self.tile_stores.values() if tst.is_smart and tst.ram_usage]:
                tile_store.refresh(onscreen)
            return            

        #calculate needed tiles
        needed_ram = 0
        base_idxs, pic_idxs, vid_idxs, cached_idxs = set(), set(), set(), set()

        #prefetch tiles that are expensive to load at a big resolution and then scale them down later
        #this saves from having to load them again and again but costs RAM
        prefetch_size = min(max(ts*PySaic.settings.PREFETCH_MULTIPLIER, self.fast_limit), self.res_limit)
        
        if ts <= PySaic.settings.BASE_TILE_RES: #tile is under or at PySaic.base_tile_res and has a length of 1
            if self.max_tile_len == 1: 
                base_idxs = blanks.copy() #skip filtering through if tiles are hard capped at len 1
            else: 
                base_idxs = set([tile_idx for tile_idx in blanks if self.tiles[tile_idx].len == 1]) 
            if ts > self.fast_limit: 
                needed_ram += len(base_idxs)*ts*ts*3

        for tile_idx in blanks.difference(base_idxs):
            tile = self.tiles[tile_idx]
            #tile is not at any sizes bigger than the current one
            if not self._find_supersized_tile(tile_idx, ts):
                if tile.source_type == "pic": pic_idxs.add(tile_idx)
                else: vid_idxs.add(tile_idx)

                if prefetch_size > self.fast_limit: 
                    needed_ram += min(tile.len, self.max_tile_len)*prefetch_size*prefetch_size*3
            #after we prefetch, we still need a tile at the current requested size, so we downscale that and essentially create 2 copies of that tile
            if ts < self.res_limit:
                cached_idxs.add(tile_idx)
                if ts > self.fast_limit: 
                    needed_ram += min(tile.len, self.max_tile_len)*ts*ts*3 
        
        if blanks:
            self.debug_strings["Cache hit ratio"] = f"{(1-round(len(vid_idxs.union(pic_idxs)) / len(blanks), 2))*100}%"

        #free ram
        self.ram_usage = sum([tile_store.ram_usage for tile_store in self.tile_stores.values()])
        freeing_stage = 0
        freeing = True
        freeable = [tst for tst in self.tile_stores.values() if tst.is_smart and tst.ram_usage]
        while freeing and freeable: 
            for tile_store in freeable:
                tile_store.refresh(onscreen) #update LRU cache statistics
                for tile_idx in tile_store.get_least_used():
                    #1 - tile is not on screen at any size - 
                    #2 - tile is on screen, but not at this size, and a higher res tile exists 
                    #3 - tile is on screen, but not at this size, and a higher res tile does not exist but tile will not be downscaled from
                    if freeing_stage == 1 and tile_idx not in onscreen \
                    or freeing_stage == 2 and (tile_store.tile_size != ts and self._find_supersized_tile(tile_idx, tile_store.tile_size)) \
                    or freeing_stage == 3 and (tile_store.tile_size != ts and tile_idx not in cached_idxs):
                        self.ram_usage -= tile_store.free_tile(tile_idx)
                    #check if we still need to free ram - this is the only line that executes at stage 0 as well
                    if self.init_ram - self.ram_usage >= needed_ram: 
                        freeing = False
                        break
            if freeing: freeing_stage += 1
            if freeing_stage == 4: break
        if freeing_stage:
            self.debug_strings["RAM status"] = STAGES[freeing_stage]
        if freeing_stage == 4:
            #stage 4 = can't free any more without destroying what's already on screen
            #streaming will allocate above what RAM can handle and at this point a numpy OOM can happen
            util.misc.log(f"Warning - unable to free {int(needed_ram - (self.init_ram - self.ram_usage))//1024/1024} megabytes of RAM for this mosaic! This may lead to a crash...")

        #if streaming is halted while executing the dispatch: return
        if not self._dispatch(base_idxs, self._base_tile, ts, "base tiles"): return
        if not self._dispatch(pic_idxs, self._pic_tile, prefetch_size, "pic tiles"): return
        
        #manually dispatch videos because they are dispatched by entire video instead of by tile
        if self.stop_streaming_flag: 
            self.prog_string = ""
            return
        self.prog_counter = 0
        vids : dict[str, list[int]] = dict() #vid path : tiles to get from vid
        for tile_idx in vid_idxs:
            tile = self.tiles[tile_idx]
            if tile.path not in vids: vids[tile.path] = []
            vids[tile.path].append(tile_idx)
        self.prog_string = f"{len(vid_idxs)} vid tiles..."
        with ThreadPoolExecutor(max(os.cpu_count()//2, 1)) as executor:
            list(executor.map(lambda vid_path, tile_idxs: self._vid_tiles(vid_path, tile_idxs, prefetch_size), vids.keys(), vids.values())) 
        
        self._dispatch(cached_idxs, self._cached_tile, ts, "cached tiles")