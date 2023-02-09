from Units_and_constants import * 
from matplotlib import cm
import taichi as ti

grid = (N_r, N_φ)

######## Prim and Cons variable fields ##########
q = ti.Vector.field(4, ti.f32, grid) # rho, u, w, p
U = ti.Vector.field(4, ti.f32, grid) # rho, γ*rho*u, γ*rho*w, rho*E

Δq_r = ti.Vector.field(4, ti.f32, grid) # slope along radius
Δq_φ = ti.Vector.field(4, ti.f32, grid) # slope along phi (polar angle)

r_L = ti.Vector.field(4, ti.f32, grid) # left face along radius interface i-½
r_R = ti.Vector.field(4, ti.f32, grid) # right face along radius interface i-½
φ_L = ti.Vector.field(4, ti.f32, grid) # left face along phi interface j-½
φ_R = ti.Vector.field(4, ti.f32, grid) # right face along phi interface j-½

rad_flux = ti.Vector.field(4, ti.f32, grid) # radius flux terms for i-½
phi_flux = ti.Vector.field(4, ti.f32, grid) # phi flux terms for j-½


######## Mass fractions and Species variable fields ##########
X = ti.Vector.field(species, ti.f32, grid)     # Mass fractions of every species in each cell
rho_X = ti.Vector.field(species, ti.f32, grid) # Species density in each cell

ΔX_r = ti.Vector.field(species, ti.f32, grid) # slope along radius
ΔX_φ = ti.Vector.field(species, ti.f32, grid) # slope along phi (polar angle)

Xr_L = ti.Vector.field(species, ti.f32, grid) # left face along radius interface i-½
Xr_R = ti.Vector.field(species, ti.f32, grid) # right face along radius interface i-½
Xφ_L = ti.Vector.field(species, ti.f32, grid) # left face along phi interface j-½
Xφ_R = ti.Vector.field(species, ti.f32, grid) # right face along phi interface j-½

rad_X_flux = ti.Vector.field(species, ti.f32, grid) # radius flux terms for i-½
phi_X_flux = ti.Vector.field(species, ti.f32, grid) # phi flux terms for j-½

# Quality of life to make the code sligthly smaller, it stores all the terms for the final divergence equations
rad_terms = ti.field(ti.f32, 4)
phi_terms = ti.field(ti.f32, 4)

X_rad_terms = ti.field(ti.f32, species)
X_phi_terms = ti.field(ti.f32, species)


######## Tim derivatives for RK2 ########
rho_X_div = ti.Vector.field(species, ti.f32, grid)
rho_X_half_timestep = ti.Vector.field(species, ti.f32, grid)

div_t = ti.Vector.field(4, ti.f32, grid) # time-derivative for each conserved variable
U_half_timestep = ti.Vector.field(4, ti.f32, grid)


######## Other variable fields for species ###########
species_m = ti.field(ti.f32, species)      # atomic mass of each species, expressed in atomic mass units u
adiabatic = ti.field(ti.f32, species)      # The adiabatic index for each species
gamma = ti.field(ti.f32, grid)             # Effective value of gamma in each cell for every iteration. Differs between cells because a difference in mass
average_m = ti.field(ti.f32, grid)         # average particle mass in each cell
T = ti.field(ti.f32, grid)                 # Temperature in each cell


dt = ti.field(ti.f32, shape=()) # For timestep variation

##### Dimensions #####
# Conversion from degrees to radians (saves on computation for later):
pi = 3.14159265359
dr = RADIUS/N_r                       # radius length of each cell
dφ = (degree_view * pi/180) / N_φ # angle φ length of each cell
fov_rad = degree_view * pi/180 # span of entire angle grid, in radians

###### 2D-Display ######
center = ti.Vector([0.5, 0.5])
index_grid = ti.Vector.field(2, int, (res, res))
pixels = ti.field(ti.f32, (res, res))
cmap = cm.get_cmap(cmap_name)
