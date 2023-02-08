import taichi as ti

@ti.kernel # precompute the cell indexes for every pixel
def assign_grid_indexes(index_grid: ti.template(), center: ti.math.vec2, zoom:float, res: int, N_r:int, N_θ: int, dθ: float, fov_rad: float):
    pi = 3.14159265359
    for i, j in index_grid:
        z = (ti.Vector([i/res, j/res]) - center) / zoom # Center the vector and scale everything
        if z.norm() * zoom < 0.5:
            r = int((z.norm()*N_r)) # get radius index
            angle_rad = ti.atan2(z.y, z.x) # get the angle [radians]. 
            if angle_rad < 0: angle_rad += 2*pi
            t = int(angle_rad / dθ) # get θ index
            if t >= 0 and t < N_θ and angle_rad >= 0 and angle_rad < fov_rad: # to fix bugs with negative indexes and angles, causing incorrect repetition and glitches
                index_grid[i, j] = [r, t] # stores indexes of cell for every pixel