import numpy as np
import numba

@numba.jit(nopython=True, nogil=True, cache=True, parallel=True, fastmath=True)
def build_matches_st(avg_colour_list_u16 : np.ndarray[np.uint16], sum_avg_sq : np.ndarray[np.uint64], 
                   source_blocks : np.ndarray[np.uint8], cmap : np.ndarray[np.int32], cpu_count : int):
    """
        single threaded version of the matching algorithm
        uses the euclidean distance formula to find the closest tile that matches a given rgb value in source_blocks

        Args:
        avg_colour_list: array of average colours of every single tile
        sum_avg_sq: array of (r^2+g^2+b^2) for every single tile
        source_blocks: 2d(!) array of rgb values from the source media that needs to be converted
        cmap: map of rgb colour from source -> closest tile index (cache to speed things up)
       
        Returns a tuple:
        first value: the converted source_blocks where each pixel has been replaced with an index into avg_colour_list
        second value: set of indices into avg_colour_list that were freshly calculated and not cached in cmap 
        (used for streaming the tiles in the renderer in later)
    """
    s_len = int(np.ceil(len(source_blocks) / cpu_count))
    segmented_match_list = np.zeros(s_len*cpu_count, dtype=np.int32)
    unique_mask = np.zeros(len(avg_colour_list_u16), dtype=np.bool_)
    
    for i in numba.prange(cpu_count):
        distances = np.zeros(len(avg_colour_list_u16))
        subslice = source_blocks[i*s_len : (i+1)*s_len]
        
        for j, colour in enumerate(subslice):
            sr, sg, sb = colour
            key = sr << 16 | sg << 8 | sb
            tile_idx = cmap[key]
            if tile_idx == -1:
                #formula = (r1^2+g1^​2+b1^2)+(r2^2+g2^​2+b2^2)−2(r1​r2+g1​g2​+b1​b2​)
                #remove (r1^2+g1^​2+b1^2) since its the same colour 
                #(r2^2+g2^​2+b2^2) for every tile is precalculated into sum_avg_sq     
                #after some testing ive found this is the fastest way to do it since numba can optimise this better than any numpy function calls   
                min_dist = np.inf
                min_idx = -1
                for k, avg_colour in enumerate(avg_colour_list_u16):
                    ar, ag, ab = avg_colour
                    distances[k] = dist = sum_avg_sq[k] - 2*(ar*sr + ag*sg + ab*sb)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = k
                cmap[key] = tile_idx = min_idx
                unique_mask[min_idx] = 1
            segmented_match_list[i*s_len + j] = tile_idx
        
    return segmented_match_list, set(np.nonzero(unique_mask)[0])

#main matching (build_match_list_1d/2d run only once, process_matches_2d has to run every new mosaic to generate new random matches)
@numba.jit(nopython=True, parallel=True, nogil=True, cache=True, fastmath=True)
def build_matches_mt_1d(avg_colour_list_u16 : np.ndarray[np.uint16], sum_avg_sq : np.ndarray[np.uint64], 
                        source_blocks : np.ndarray[np.uint8], cpu_count : int):
    """
        matches all source rgb values to their closest match in avg_colour_list_u16

        Args:
        avg_colour_list_u16: array of average colours of every single tile
        sum_avg_sq: array of (r1^2+g1^2+b1^2) for every single tile
        source_blocks: 2d(!) array of rgb values from the source media that needs to be converted
        cpu_count: no. of threads to use for the processing
       
        Returns:
        converted version of source_blocks where all rgb values are replaced with indices into avg_colour_list_u16
    """
    s_len = int(np.ceil(len(source_blocks) / cpu_count))
    segmented_match_list = np.zeros((cpu_count, s_len), dtype=np.int32)
    cmap = np.zeros(2**24, dtype=np.int32)
    cmap.fill(-1)
    
    for i in numba.prange(cpu_count):
        distances = np.zeros(len(avg_colour_list_u16))
        subslice = source_blocks[i*s_len : (i+1)*s_len]
        submatch_list = np.zeros(s_len, dtype=np.int32)
        
        for j, colour in enumerate(subslice):
            sr, sg, sb = colour
            key = sr << 16 | sg << 8 | sb
            tile_idx = cmap[key]
            if tile_idx == -1:
                for k, avg_colour in enumerate(avg_colour_list_u16):
                    ar, ag, ab = avg_colour
                    distances[k] = sum_avg_sq[k] - 2*(ar*sr + ag*sg + ab*sb)
                cmap[key] = tile_idx = np.argmin(distances)
            submatch_list[j] = tile_idx
        segmented_match_list[i] = submatch_list
        
    return segmented_match_list.reshape(-1)

@numba.jit(nopython=True, parallel=True, nogil=True, cache=True, fastmath=True)
def build_matches_mt_2d(avg_colour_list_u16 : np.ndarray[np.uint16], sum_avg_sq : np.ndarray[np.uint64], 
                        source_blocks : np.ndarray[np.uint8], cpu_count : int, random_choice : int):
    """
        matches all source rgb values to len(random_choice) number of closest matches in avg_colour_list_u16  
        
        this is one half of the dithering 'algorithm' used, where we select len(random_choice) number of closest matches  
        and then in process_matches_2d we randomly select one of those matches to display  
         this introduces noise in the final render which can break up colour banding and make up for the quantisation

        Args:
         avg_colour_list_u16: array of average colours of every single tile
         sum_avg_sq: array of (r1^2+g1^2+b1^2) for every single tile
         source_blocks: 2d(!) array of rgb values from the source media that needs to be converted
         cpu_count: no. of threads to use for the processing
         random_choice: how many tile indices to save for a different colour
       
        Returns:
         converted version of source_blocks where all rgb values with an array of len(random_choice)
         that contains a selection of the closest sorted matches to that pixel 
    """
    s_len = int(np.ceil(len(source_blocks) / cpu_count))
    segmented_match_list = np.zeros((cpu_count, s_len, random_choice), dtype=np.int32) 
    cmap = np.zeros((2**24, random_choice), dtype=np.int32)
    cmap.fill(-1)
    
    for i in numba.prange(cpu_count):
        distances = np.zeros(len(avg_colour_list_u16))
        subslice = source_blocks[i*s_len : (i+1)*s_len]
        submatch_list = np.zeros((s_len, random_choice), np.int32) 
        
        for j, colour in enumerate(subslice):
            sr, sg, sb = colour
            key = sr << 16 | sg << 8 | sb
            tile_idxs = cmap[key]

            if tile_idxs[0] == -1:
                for k, avg_colour in enumerate(avg_colour_list_u16):
                    ar, ag, ab = avg_colour
                    distances[k] = sum_avg_sq[k] - 2*(ar*sr + ag*sg + ab*sb)
                parted = np.argpartition(distances, random_choice)[:random_choice] #get the smallest x values
                cmap[key] = tile_idxs = parted[np.argsort(distances[parted])].astype(np.int32) #sorting shenanigans
                
            submatch_list[j] = tile_idxs
        segmented_match_list[i] = submatch_list
    
    return segmented_match_list.reshape(-1, random_choice)

@numba.jit(nopython=True, nogil=True, cache=True)
def process_matches_2d(match_arr : np.ndarray[np.ndarray[np.int32]], random_choice : int): 
    """
        takes a 2d match_array and returns a 1d version where a random match is selected from each sub-array
        
        this is the second half of the dithering 'algorithm' used, 
        where in build_matches_mt_2d we select len(random_choice) number of closest matches  
        and then here we randomly select one of those matches to display  
         this introduces noise in the final render which can break up colour banding and make up for the quantisation

        Args:
         match_arr: 2d array of tile indices, where each value is an array of len(random_choice) potential matches for that pixel
         random_choice: how many tile indices were selected for a different colour
         (can be different from how many are actually in the array to limit the selection to closer matches)
    """
    fm_list = np.zeros(len(match_arr), dtype=np.int32)
    for i, match in enumerate(match_arr):
        fm_list[i] = match[np.random.randint(0, random_choice) if random_choice else 0]
    return fm_list