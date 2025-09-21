def comp_2d(a, b):
    return a[0] == b[0] and a[1] == b[1]

def comp_3d(a, b):
    return (a[0] == b[0] and 
            a[1] == b[1] and
            a[2] == b[2])

def check_point_in_bounds(tl, pos, br):
    return (tl[0] <= pos[0] <= br[0]
        and tl[1] <= pos[1] <= br[1])

def check_points_in_bounds(points : list[int], tl, br):
    for point in points:
        if not check_point_in_bounds(tl, point, br):
            return False
    return True