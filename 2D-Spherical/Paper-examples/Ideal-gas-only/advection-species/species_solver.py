import taichi as ti
from classes import *
import functions as mole

@ti.kernel
def calc_averages():
    for i, j in X:
        num_particles = 0.
        for k in ti.static(range(species)):
            particles_per_mass = X[i, j][k] / species_m[k]
            ti.atomic_add(num_particles, particles_per_mass)
        average_m[i, j] = 1 / num_particles # Actually the density is in here but its factored out.
    
        sum = 0.
        for k in ti.static(range(species)):
            term = (1 / (adiabatic[k] - 1.)) * X[i, j][k] / species_m[k]
            ti.atomic_add(sum, term)
        gamma[i, j] = 1 / (average_m[i, j] * sum) + 1


@ti.func
def species_slopes(i, j):
    R, L = mole.boundary_c(i, N_r)
    for k in ti.static(range(species)):
        ΔX_r[i, j][k] = mole.minmod(X[i, j][k] - X[L, j][k], X[R, j][k] - X[i, j][k])/dr # Uses minmode slope limiter
    R, L = mole.boundary_c(j, N_φ)
    for k in ti.static(range(species)):
        ΔX_φ[i, j][k] = mole.minmod(X[i, j][k] - X[i, L][k], X[i, R][k] - X[i, j][k])/dφ

@ti.func
def species_faces(i, j):
    # calculate faces (left and right)
    R, L = mole.boundary_c(i, N_r)
    Xr_L[i, j] = X[L, j] + (dr/2)*ΔX_r[L, j]
    Xr_R[i, j] = X[i, j] - (dr/2)*ΔX_r[i, j]
    R, L = mole.boundary_c(j, N_φ)
    Xφ_L[i, j] = X[i, L] + (dφ/2)*ΔX_φ[i, L]
    Xφ_R[i, j] = X[i, j] - (dφ/2)*ΔX_φ[i, j]

@ti.func
def species_riemman(i, j,radius_L: ti.math.vec4, radius_R: ti.math.vec4, phi_L: ti.math.vec4,  phi_R: ti.math.vec4, gamma):
    for k in ti.static(range(species)):
        rad_X_flux[i, j][k] = species_flux(0, radius_L, radius_R, Xr_L[i, j][k], Xr_R[i, j][k], gamma[i, j])
        phi_X_flux[i, j][k] = species_flux(1, phi_L   ,    phi_R, Xφ_L[i, j][k], Xφ_R[i, j][k], gamma[i, j])

@ti.func
def divergence(i, j, r_plus, r_minus, r):
    # Radius terms:
    R, L = mole.boundary_c(i, N_r)
    for k in ti.static(range(species)):
        X_rad_terms[k]= (1/r**2) * ((r_plus**2 * rad_X_flux[R, j][k]) - (r_minus**2 * rad_X_flux[i, j][k])) / dr

    # Phi terms:
    R, L = mole.boundary_c(j, N_φ)
    for k in ti.static(range(species)):
        X_phi_terms[k] = (1/r) * (phi_X_flux[i, R][k] - phi_X_flux[i, j][k]) / dφ

    # Complete time derivative for conserved species
    for k in ti.static(range(species)):
        rho_X_div[i, j][k] = - (X_rad_terms[k] + X_phi_terms[k])

@ti.func
def species_flux(Axis, Left_state: ti.math.vec4, Right_state: ti.math.vec4, X_l, X_r, gamma):
    rho_l, u_l, w_l, P_l =  Left_state[0],  Left_state[1],  Left_state[2],  Left_state[3]
    rho_r, u_r, w_r, P_r = Right_state[0], Right_state[1], Right_state[2], Right_state[3]

    # compute star (averaged) states
    rho_star = 0.5*(rho_l + rho_r)
    u_star   = 0.5*(u_l + u_r)
    w_star   = 0.5*(w_l + w_r)
    X_star   = 0.5*(X_l + X_r)

    left_axis_velocity, right_axis_velocity = 0., 0.
    flux_rho_X = 0.

    # fluxes
    if Axis == 0: # flux for radial terms
        flux_rho_X = rho_star * X_star * u_star
        left_axis_velocity = u_l
        right_axis_velocity = u_r

    elif Axis == 1: # flux for phi terms
        flux_rho_X = rho_star * X_star * w_star
        left_axis_velocity = w_l
        right_axis_velocity = w_r

    # Wavespeeds
    c_l = ti.sqrt(gamma*P_l/rho_l) + ti.abs(left_axis_velocity)
    c_r = ti.sqrt(gamma*P_r/rho_r) + ti.abs(right_axis_velocity)
    c = ti.max(c_l, c_r)

    # add stabilizing diffusive term
    flux_rho_X   -= c * 0.5 * (rho_r * X_r - rho_l * X_l)

    return flux_rho_X