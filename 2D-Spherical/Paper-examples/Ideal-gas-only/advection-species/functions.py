import taichi as ti
from classes import C, gamma, c_light, N_r, N_φ, dr, dφ, degree_view, q, U, dt

########### Hydrodynamics #############
@ti.func
def get_radius(i):
    return (dr/2) + (i*dr) # dont return zero

@ti.func
def boundary_c(b, N): 
    R = b+1 # go right
    if R > N-1: R = N-1 # dont step over the grid
    L = b-1 # go left
    if L < 0: L = 0
    
    if degree_view == 360 and N == N_φ: # make periodic when the grid spans 360 degrees
        R = b+1
        if R > N-1: R = 0
        L = b-1
        if L < 0: L = N-1
    return R, L

@ti.func
def minmod(a, b): # Minmode slope limiter! inputs are: a == q[i] - q[L], b == q[R] - q[i]
    Δ = 0.0
    if ti.abs(a) < ti.abs(b) and a*b > 0:
        Δ = a
    if ti.abs(a) > ti.abs(b) and a*b > 0:
        Δ = b
    return Δ
    


########### Relativity ################

@ti.func
def lorentz(velocity): # lorentz factor
    a = (velocity / c_light)**2
    return 1/(ti.sqrt(1 - a))

@ti.func
def get_velocities_from_momentum(p_u, p_w, rho): # input is the momentum components and the density
    # Page 2, https://arxiv.org/pdf/1806.08680.pdf
    total_momentum = ti.sqrt(p_u**2 + p_w**2)
    relativistic_total_energy = ti.sqrt(rho**2 * c_light**4 + total_momentum**2 * c_light**2)
    u = c_light**2 * p_u / relativistic_total_energy
    w = c_light**2 * p_w / relativistic_total_energy
    return u, w

@ti.func
def get_kinetic_density(vel):
    kinetic_density = 0.
    if vel != 0 and vel < 0.001 * c_light: kinetic_density = vel**2 / 2 # Enforce newtonian limit if its too small
    else: kinetic_density = (lorentz(vel) - 1.) * c_light**2 # relativistic kinetic energy (density)
    return kinetic_density

######### Miscellaneous ###################

@ti.kernel
def get_max_vel() -> ti.f32:
    max_vel = 0.
    for i, j in q:
        u, w = q[i, j][1], q[i, j][2]
        vel = ti.sqrt(u**2 + w**2)

        ti.atomic_max(max_vel, vel)
    return max_vel


@ti.kernel
def check_if_conserved() -> ti.math.vec4:
    tot_mass = 0.
    tot_rad_mom = 0.
    tot_phi_mom = 0.
    tot_rho_E = 0.
    for i, j in q:
        if i != 0 and i != 1 and i!=N_r-1 and i!=N_r-2: # Dont count ghost cells -> some mass/energy dissappears at edges
            r_inner = get_radius(i-0.5) # Radius of inner sphere (basicly radius at interface i-½)
            vol = (2/3)*((r_inner+dr)**3 - r_inner**3) * dφ # volume = (2/3) * [(r+Δr)³ - r³] * Δφ == volume of outer wedge (with radius=r+Δr) minus volume of inner wedge (with radius=r)
            # Amount of conserved quantity inside each cell:
            mass     = q[i, j][0] * vol
            rad_mom  = U[i, j][1] * vol 
            phi_mom  = U[i, j][2] * vol
            rho_E    = U[i, j][3] * vol # energy per unit volume * volume
            ti.atomic_add(tot_mass, mass)
            ti.atomic_add(tot_rad_mom, rad_mom)
            ti.atomic_add(tot_phi_mom, phi_mom)
            ti.atomic_add(tot_rho_E, rho_E)
    return ti.math.vec4(tot_mass, tot_rad_mom, tot_phi_mom, tot_rho_E)