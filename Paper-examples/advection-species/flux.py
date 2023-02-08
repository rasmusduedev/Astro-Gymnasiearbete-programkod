import functions as mole
import taichi as ti


@ti.func
def riemann_flux(Axis, Left_state: ti.math.vec4, Right_state: ti.math.vec4, gamma):
    rho_l, u_l, w_l, P_l =  Left_state[0],  Left_state[1],  Left_state[2],  Left_state[3]
    rho_r, u_r, w_r, P_r = Right_state[0], Right_state[1], Right_state[2], Right_state[3]

    # compute star (averaged) states
    rho_star = 0.5*(rho_l + rho_r)
    u_star   = 0.5*(u_l + u_r)
    w_star   = 0.5*(w_l + w_r)
    P_star   = 0.5*(P_l + P_r)

    e_l = P_l / (rho_l * (gamma-1.))
    e_r = P_r / (rho_r * (gamma-1.))
    vel_l = ti.sqrt(u_l**2 + w_l**2)
    vel_r = ti.sqrt(u_r**2 + w_r**2)
    kinetic_l = mole.get_kinetic_density(vel_l)
    kinetic_r = mole.get_kinetic_density(vel_r)
    E_l = e_l + kinetic_l
    E_r = e_r + kinetic_r

    E_star = 0.5*(E_l + E_r)

    left_axis_velocity, right_axis_velocity = 0., 0.
    flux_rho, flux_mom_u, flux_mom_w, flux_rho_E = 0., 0., 0., 0.

    vel = ti.sqrt(u_star**2 + w_star**2)
    # fluxes (local Lax-Friedrichs/Rusanov) - modified by me
    if Axis == 0: # flux for radial terms
        flux_rho   = rho_star * u_star
        flux_mom_u = mole.lorentz(vel) * rho_star * u_star**2
        flux_mom_w = mole.lorentz(vel) * rho_star * u_star * w_star
        flux_rho_E = (rho_star * E_star + P_star) * u_star

        left_axis_velocity = u_l
        right_axis_velocity = u_r

    elif Axis == 1: # flux for phi terms
        flux_rho   = rho_star * w_star
        flux_mom_u = mole.lorentz(vel) * rho_star * u_star * w_star 
        flux_mom_w = mole.lorentz(vel) * rho_star * w_star**2 + P_star # Put the pressure back in the flux again, removed expanded aximuthal pressure-gradient term in the azimuthal momentum to reverse the change
        flux_rho_E = (rho_star * E_star + P_star) * w_star

        left_axis_velocity = w_l
        right_axis_velocity = w_r

    # Wavespeeds
    c_l = ti.sqrt(gamma*P_l/rho_l) + ti.abs(left_axis_velocity)
    c_r = ti.sqrt(gamma*P_r/rho_r) + ti.abs(right_axis_velocity)
    c = ti.max(c_l, c_r)

    # add stabilizing diffusive term
    flux_rho   -= c * 0.5 * (rho_r - rho_l)
    flux_mom_u -= c * 0.5 * (mole.lorentz(vel_r) * rho_r * u_r - mole.lorentz(vel_l) * rho_l * u_l) # notice the relativistic momentum
    flux_mom_w -= c * 0.5 * (mole.lorentz(vel_r) * rho_r * w_r - mole.lorentz(vel_l) * rho_l * w_l)
    flux_rho_E -= c * 0.5 * (rho_r * E_r - rho_l * E_l)

    return ti.Vector([flux_rho, flux_mom_u, flux_mom_w, flux_rho_E])