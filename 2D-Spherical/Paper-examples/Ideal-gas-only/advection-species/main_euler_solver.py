import functions as mole
from classes import *
import taichi as ti
import flux as flu
import species_solver as nuk


@ti.kernel
def set_BC(): # reflective boundary conditions at core AND at surface
    for i, j in U:
        U[0, j] = U[3, j]
        U[1, j] = U[2, j]
        U[N_r-1, j] = U[N_r-4, j]
        U[N_r-2, j] = U[N_r-3, j]

        # Let momentum go towards zero when it approaches the core (and surface)... these are reflective boundary conditions
        U[0, j][1] = -U[3, j][1]
        U[1, j][1] = -U[2, j][1]
        U[N_r-1, j][1] = -U[N_r-4, j][1]
        U[N_r-2, j][1] = -U[N_r-3, j][1]

@ti.kernel
def calc_dt():
    dt[None] = 1.0e5 # arbitrarily large number
    for i, j in q:
        rho, u, w, P = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        r = mole.get_radius(i)
        c = ti.sqrt(gamma[i, j] * P / rho)
        welp_rad = C * dr / (ti.abs(u) + c)
        welp_phi = C * dφ*r / (ti.abs(w) + c) 
        welp = ti.min(welp_rad, welp_phi)
        ti.atomic_min(dt[None], welp)
    # Stores new timestep value in dt[None]




@ti.kernel
def cons_to_prim(U_field: ti.template(), rho_X_field: ti.template()): # Either U_half_timestep or U to be converted...
    for i, j in U:
        rho, p_u, p_w, rho_E = U_field[i, j][0], U_field[i, j][1], U_field[i, j][2], U_field[i, j][3]
        q[i, j][0] = rho
        q[i, j][1], q[i, j][2] = mole.get_velocities_from_momentum(p_u, p_w, rho)

        vel = ti.sqrt(q[i, j][1]**2 + q[i, j][2]**2)
        kinetic_density = mole.get_kinetic_density(vel)
        e = (rho_E / rho) - kinetic_density
        q[i, j][3] = rho * e * (gamma[i, j] - 1.)

        X[i, j] = rho_X_field[i, j] / rho
        T[i, j] = q[i, j][3] * average_m[i, j] * u_in_mass_units / (rho * k_B)
        

@ti.kernel
def prim_to_cons():
    for i, j in q:
        rho, u, w, P = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        U[i, j][0] = rho
        vel = ti.sqrt(u**2 + w**2)
        U[i, j][1] = mole.lorentz(vel) * rho * u # relativistic momentum
        U[i, j][2] = mole.lorentz(vel) * rho * w

        kinetic_density = mole.get_kinetic_density(vel)
        e = P / (rho * (gamma[i, j] - 1.))
        E = e + kinetic_density
        U[i, j][3] = rho * E

        rho_X[i, j] = rho * X[i, j]


@ti.kernel
def all(result_div: ti.template()):
    for i, j in q:
        R, L = mole.boundary_c(i, N_r)
        for k in ti.static(range(4)):
            Δq_r[i, j][k] = mole.minmod(q[i, j][k] - q[L, j][k], q[R, j][k] - q[i, j][k])/dr # Uses minmode slope limiter
        R, L = mole.boundary_c(j, N_φ)
        for k in ti.static(range(4)):
            Δq_φ[i, j][k] = mole.minmod(q[i, j][k] - q[i, L][k], q[i, R][k] - q[i, j][k])/dφ
        nuk.species_slopes(i, j)
    for i, j in q:
        # calculate faces (left and right)
        R, L = mole.boundary_c(i, N_r)
        r_L[i, j] = q[L, j] + (dr/2)*Δq_r[L, j]
        r_R[i, j] = q[i, j] - (dr/2)*Δq_r[i, j]
        R, L = mole.boundary_c(j, N_φ)
        φ_L[i, j] = q[i, L] + (dφ/2)*Δq_φ[i, L]
        φ_R[i, j] = q[i, j] - (dφ/2)*Δq_φ[i, j]
        nuk.species_faces(i, j)
    for i, j in q:
        # Left and right states along each axis
        radius_L, radius_R = r_L[i, j], r_R[i, j]
        phi_L ,  phi_R = φ_L[i, j], φ_R[i, j]

        # radius axis flux and azimuth axis/plane flux:
        rad_flux[i, j] = flu.riemann_flux(0, radius_L, radius_R, gamma[i, j])
        phi_flux[i, j] = flu.riemann_flux(1,    phi_L,    phi_R, gamma[i, j])
        nuk.species_riemman(i, j, radius_L, radius_R, phi_L, phi_R, gamma)
    for i, j in q:
        r = mole.get_radius(i)
        r_minus = mole.get_radius(i-0.5) # radius at i-½ applied to the flux at that interface (given by flux[i, j])
        r_plus  = mole.get_radius(i+0.5) # radius at i+½ applied to the flux at that interface (giver by flux[R, j])
        
        # Radius terms:
        R, L = mole.boundary_c(i, N_r)
        for k in ti.static(range(4)):
            rad_terms[k]= (1/r**2) * ((r_plus**2 * rad_flux[R, j][k]) - (r_minus**2 * rad_flux[i, j][k])) / dr

        # Phi terms:
        R, L = mole.boundary_c(j, N_φ)
        for k in ti.static(range(4)):
            phi_terms[k] = (1/r) * (phi_flux[i, R][k] - phi_flux[i, j][k]) / dφ

        rho, u, w, P = q[i, j][0], q[i, j][1], q[i, j][2], q[i, j][3]
        R, L = mole.boundary_c(i, N_r)
        dP_dr = (q[R, j][3] - q[i, j][3])/dr # extra pressure derivative term appearing in the radial momentum

        # Complete time derivatives for conserved variables
        result_div[i, j][0] = - (rad_terms[0] + phi_terms[0])
        result_div[i, j][1] = - (rad_terms[1] + phi_terms[1]) - dP_dr      + (rho*w**2)/r
        result_div[i, j][2] = - (rad_terms[2] + phi_terms[2]) - (rho*u*w)/r 
        result_div[i, j][3] = - (rad_terms[3] + phi_terms[3])

        nuk.divergence(i, j, r_plus, r_minus, r)

@ti.kernel
def euler_add(a_timestep: ti.template(), b: ti.template(), timestep: float, dev_n: ti.template()):
    for i, j in b:
        a_timestep[i, j] = b[i, j] + timestep * dev_n[i, j]

# The RK2 solving scheme
def Euler_scheme():
    set_BC()

    # NOTE: Integrate for half timestep!
    nuk.calc_averages()
    cons_to_prim(U, rho_X) # From U^n to q^n
    calc_dt()
    all(div_t) # get time-slope at U^n [using q^n]
    euler_add(U_half_timestep, U, dt[None]/2, div_t) # Euler step to half timestep
    euler_add(rho_X_half_timestep, rho_X, dt[None]/2, rho_X_div)

    # NOTE: Integrate for whole timestep!
    nuk.calc_averages()
    cons_to_prim(U_half_timestep, rho_X_half_timestep) # From U^n+½ to q^n+½
    all(div_t) # get slope at U^n+½ (half timestep) [using q^n+½]
    euler_add(U, U, dt[None], div_t) # Back to U^n and take Euler step to full timestep (with slope from n+½)
    euler_add(rho_X, rho_X, dt[None], rho_X_div)