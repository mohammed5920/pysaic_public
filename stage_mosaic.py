from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os

import numpy as np

from pysaic import PySaic, Stage
import util.misc
import mosaic.source
import mosaic.tiles
import mosaic.match

#mosaic processing stage (does not draw any UI elements to the stage)
#serves as the bridge between stage_main and stage_render
#the metadata is processed into a valid mosaic and stored in PySaic.mosaic

class Mosaic(Stage):
    def analyse_tiles(self, func, tile_paths : list[str], cpu_count, nested=False) -> list[mosaic.tiles.Tile]:
        """distribute tile loading to all cores
        \nargs:
        func: reference to which function to distribute
        tile_paths: paths to the tiles needing to be distributed
        nested: whether or not results need to be flattened before returning
        """
        result = []
        with tqdm(total=len(tile_paths)) as PySaic.pbar:
            with ThreadPoolExecutor(cpu_count) as executor:
                #trim all Nones and Falses and raise any exceptions
                if nested: [result.extend(tile) for tile in executor.map(func, tile_paths) if tile]
                else: [result.append(tiles) for tiles in executor.map(func, tile_paths) if tiles]
                return result 
            
    def filter_tiles(self, tiles : list[mosaic.tiles.Tile]) -> list[mosaic.tiles.Tile]:
        """filters out tiles that have the same colour"""
        tile_set = set()
        filtered_tiles = []
        for tile in tiles:
            if tile.colour_key in tile_set:
                continue
            tile_set.add(tile.colour_key)
            filtered_tiles.append(tile)
        print(f"Filtered {((len(tiles) - len(filtered_tiles)) / len(tiles)) * 100:.2f}% of tiles.")
        return filtered_tiles

    def process_mosaic(self):
        """takes all the metadata stored in PySaic.temp
        and processes it into Pysaic.mosaic"""
        
        os.system("cls")
        if PySaic.mosaic.mode == "vid": PySaic.mosaic.source.kill_threads() #since we're going to be messing with the global vars it reads from

        # ---------------- step 0 -> declare our key players ---------------- 
        # ///////////////////////////////// tiles ///////////////////////////

        tiles : list[mosaic.tiles.Tile] = []
        unique_tiles : set[int] = None #set of unique tile indices used in the mosaic
        #in the case of video, its added to dynamically as the video streams in

        #this is uint16 instead of uint8 because colours are multiplied together during matching
        avg_colours : np.ndarray[np.ndarray[np.uint16]] = None 
    
        #the tiles are matched according to the minimum euclidian distance between an rgb pixel of the source
        #and the average rgb colour of every tile
        #formula is sqrt( (r1 - r2)^2 + (g1 - g2)^2 + (b1 - b2)^2 )
        #expanding the brackets we get (assume ^ = ^2 for brevity)
        #(r1^+g1^​+b1^)+(r2^+g2^​+b2^)−2(r1​r2+g1​g2​+b1​b2​)
        #since during matching, the source rgb pixels change but the tiles are always the same
        #we can precalculate (r^+g^​+b^) for every single tile
        sum_avg_sq : np.ndarray[np.uint64] = None  #stores an array where each value is the sum of (r1^2+g1^2​+b1^2) of the tile's average colour 

        # /////////////////////////// source ///////////////////////////

        source : mosaic.source.PicSource | mosaic.source.VidSource = None
        rebuild_match_flag = False #whether or not to build/rebuild the matches between source and tile rgb values 

        # ////////////////////////// matching //////////////////////////

        matches : np.ndarray[np.int32] = None #no random enabled

        #we add 'dithering' by finding len(PySaic.temp.rand_choice) number of closest matches  
        #and then randomly selecting one of those matches to display  
        raw_matches : np.ndarray[np.ndarray[np.int32]] | np.ndarray[np.int32] = None #random may be enabled

        # ///////////////////////// rendering /////////////////////////

        #if the user chooses a folder with 10k tiles,
        #but an image that only uses 3 of those tiles,
        #the way the matching works is based on indices into the big list with those 10k tiles
        #if one of those 3 tiles is tile no. 10k or some other large number, to stream in and display that tile
        # we will have to allocate an array of alteast that length, allocating ram that cannot be used otherwise
        #                                 so instead 
        #we create an array where each index = an index into tiles (from the match list)
        #and each value is an index into the arrays used to store the streamed in tile textures
        #this saves on committed RAM (in resource monitor) which stops oom exceptions from being raised
        tile_map : np.ndarray[np.int32] = None 

        tile_len_map : np.ndarray[np.int32] = None #idx: tile index from matches -> value: length of tile
        mean_tile_len : int = None #avg length of tiles used in mosaic
        max_tile_len : int = None #max length a tile used in mosaic can be 
        #this is used in tile streaming, avg length is used to guess RAM usage 
        #max length is used to cap where moving tiles aren't needed, reduces CPU and RAM load
        #in cases where avg > max, max is also for RAM predictions

        total_frames : int = None #sum of no. of tiles and the no. of frames in each tile

        # ---------------- step 1 -> analyse tiles ---------------- 
        # reanalyse if tile folders have changed or media type has been enabled/disabled
        
        print("Analysing tiles... ")
        start = time.perf_counter()
        if PySaic.temp.enable_pics:
            if (PySaic.mosaic.tile_folder != PySaic.temp.tile_folder or not PySaic.mosaic.enable_pics): 
                print("Analysing pictures...")
                tiles.extend(self.analyse_tiles(mosaic.tiles.pic_tile, PySaic.temp.pic_tiles, os.cpu_count()))
                rebuild_match_flag = True 
            else:
                tiles.extend([tile for tile in PySaic.mosaic.tiles if tile.source_type == "pic"])
        if PySaic.temp.enable_vids:
            if (PySaic.mosaic.tile_folder != PySaic.temp.tile_folder or not PySaic.mosaic.enable_vids):
                print("Analysing videos...")
                tiles.extend(self.analyse_tiles(mosaic.tiles.video_tiles, PySaic.temp.vid_tiles, os.cpu_count()//2, nested=True))
                rebuild_match_flag = True
            else:
                tiles.extend([tile for tile in PySaic.mosaic.tiles if tile.source_type == "vid"])
        
        #since this is all happening on a separate thread i need some way to cancel if user hits ESCAPE
        #easiest way is to just check this flag before every major processing step
        if PySaic.stop_loading_flag: return

        if not tiles: #all the tiles have raised exceptions (very rare but does happen)
            PySaic.UI.toast("No tiles could be loaded! Try choosing another folder or enabling all media types...")
            PySaic.stop_loading_flag = True #hack to go back home
            return

        rebuild_match_flag = rebuild_match_flag or (PySaic.temp.enable_pics != PySaic.mosaic.enable_pics
                                                or  PySaic.temp.enable_vids != PySaic.mosaic.enable_vids)
        
        if rebuild_match_flag: #refilter the tiles if the state of the tiles have changed
            tiles = self.filter_tiles(tiles)
            avg_colours = np.array([tile.avg_colour for tile in tiles], dtype=np.uint16)  
            sum_avg_sq = np.sum(avg_colours ** 2, axis=1) 
        else:
            avg_colours = PySaic.mosaic.colours
            sum_avg_sq = PySaic.mosaic.clr_sq_sum
        print(f"done in {time.perf_counter() - start} seconds - {len(tiles)} tiles loaded")
        
        # ---------------- step 2 -> analyse source ---------------- 

        if PySaic.temp.source_path != PySaic.mosaic.source_path \
        or PySaic.mosaic.source.dscale_factor != np.ceil(np.max(np.divide(PySaic.mosaic.source.raw_size, PySaic.resolution))):
            if PySaic.temp.mode != "vid": 
                source = mosaic.source.PicSource(PySaic.temp.source_path, PySaic.resolution)
            else: 
                source = mosaic.source.VidSource(PySaic.temp.source_path, PySaic.resolution)
            rebuild_match_flag = True
        else:
            source = PySaic.mosaic.source
        if PySaic.stop_loading_flag: return

        # ---------------- step 3 -> match ----------------  
        # rebuild matches if tiles have changed, source has changed, or user wants more dithering 

        rebuild_match_flag = rebuild_match_flag or PySaic.temp.rand_choice > PySaic.mosaic.rand_choice
        
        if PySaic.temp.mode != "vid" and rebuild_match_flag:
            print("Building match list... ", end="")
            if PySaic.stop_loading_flag: return

            start = time.perf_counter()
            if not PySaic.temp.rand_choice:
                raw_matches = mosaic.match.build_matches_mt_1d(avg_colours, sum_avg_sq, source.blocks, os.cpu_count())
            else:
                raw_matches = mosaic.match.build_matches_mt_2d(avg_colours, sum_avg_sq, source.blocks, os.cpu_count(), PySaic.temp.rand_choice)
            print(f"done in {time.perf_counter() - start} seconds.")
        else:
            raw_matches = PySaic.mosaic.raw_matches
        
        #the multithreaded matching process can add an arbitrary amount of zeroes to the end of the matched array
        #this messes with the renderer so we cap it here
        if PySaic.temp.mode != "vid" and raw_matches.shape[0] != source.blocks.shape[0]:
            assert raw_matches.shape[0] > source.blocks.shape[0]
            raw_matches = raw_matches[:source.blocks.shape[0]]

        if PySaic.stop_loading_flag: return

        #step 4 -> process the match array further to flatten any random matches and extract the unique tile indices used 
        if PySaic.temp.mode != "vid":
            print(f"Processing {len(raw_matches)} matches... ", end="")
            start = time.perf_counter()
            if np.isscalar(raw_matches[0]):
                matches = raw_matches
            else:
                matches = mosaic.match.process_matches_2d(raw_matches, PySaic.temp.rand_choice)
            unique_tiles = set(matches)
            if PySaic.stop_loading_flag: return
            print(f"done in {time.perf_counter() - start} seconds.")
        else:
            source.reset()
            raw_matches, matches = [None]*2 #streamed in dynamically 
            unique_tiles = source.unique_tiles #streamed in dynamically

        #step 5 -> create the various maps used for optimised rendering and streaming
        #in the case of video we assume worst case and prepare to stream in all discovered tiles
        print("Generating rendering maps...", end="")
        start = time.perf_counter()
        streamable = unique_tiles if PySaic.temp.mode != "vid" else range(len(tiles))
        map_size = np.max(matches)+1 if PySaic.temp.mode != "vid" else len(tiles)

        tile_map = np.zeros(map_size, dtype=np.int32) #index: id from matches -> new id with no gaps
        tile_len_map = np.zeros(map_size, dtype=np.int32) #index: id from matches -> how many frames in tile 
        
        tile_lens = [tiles[tile_idx].len for tile_idx in streamable]
        mean_tile_len = int(np.mean(tile_lens))
        if PySaic.temp.mode == "hyb":
            if PySaic.settings.CAP_TILE_LENGTHS:
                max_tile_len = mean_tile_len
            else:
                max_tile_len = max(tile_lens)
        else:
            max_tile_len = 1
    
        offset = 0
        for tile_idx in streamable: 
            tile_map[tile_idx] = offset
            frame_count = min(tiles[tile_idx].len, max_tile_len)
            tile_len_map[tile_idx] = frame_count
            offset += frame_count
        total_frames = offset
        print(f"done in {time.perf_counter() - start} seconds.")

        #step 6 -> commit to global mosaic vars 
        #this is the ONLY place in the entire 3000 lines or so of this script where these variables are set 
        PySaic.mosaic.clone(PySaic.temp) #clone all user settings
        PySaic.mosaic.source = source

        PySaic.mosaic.tiles, PySaic.mosaic.unique_tiles = tiles, unique_tiles
        PySaic.mosaic.colours, PySaic.mosaic.clr_sq_sum = avg_colours, sum_avg_sq
        
        PySaic.mosaic.raw_matches, PySaic.mosaic.matches = raw_matches, matches
        PySaic.mosaic.tile_map, PySaic.mosaic.tile_len_map = tile_map, tile_len_map
        PySaic.mosaic.mean_tile_len, PySaic.mosaic.max_tile_len = mean_tile_len, max_tile_len
        PySaic.mosaic.total_frames = total_frames

    def start(self):
        if not os.path.exists("caches/"): os.mkdir("caches/")
        PySaic.loading_thread = util.misc.ThreadWrapper(target=self.process_mosaic)
        
        def returnfrom_processmosaic(self):
            PySaic.pbar.close()
            if PySaic.stop_loading_flag: #nothing changed in global vars
                PySaic.stop_loading_flag = False
                if not PySaic.return_stage():
                    PySaic.switch_stage("main")
            else:
                PySaic.switch_stage("render") 

        PySaic.transfer_stage("loading", returnfrom_processmosaic)