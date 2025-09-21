import numba
import numpy as np

def crop(array : np.ndarray, copy=True):
    """returns (a copy of) the input image array cropped to a square and centred"""
    min_dim = min(array.shape[:2])
    start_y = (array.shape[0] - min_dim) // 2
    start_x = (array.shape[1] - min_dim) // 2
    result = array[start_y : start_y + min_dim,
                 start_x : start_x + min_dim]
    return result if not copy else result.copy()

@numba.jit(nopython=True, cache=True, nogil=True)
def mse(a : np.ndarray[np.uint16], b : np.ndarray[np.uint16]) -> np.ndarray[np.uint64]:
    """returns the MSE between 2 colours"""
    ar, ag, ab = a
    br, bg, bb = b
    cr, cg, cb = ar-br, ag-bg, ab-bb 
    return (cr**2 + cg**2 + cb**2)/3

@numba.jit(nopython=True, cache=True, nogil=True)
def rgb_mean(rgb_array: np.ndarray[np.ndarray[np.uint8]]):
    """calculates the average colour of an rgb array"""
    rgb = np.zeros(3, np.int64)
    iters = 0
    for row in rgb_array:
        for colour in row:
            rgb += colour
            iters += 1
    return rgb / iters