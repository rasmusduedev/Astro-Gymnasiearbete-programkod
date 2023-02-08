import main_euler_solver as euler
import species_solver as nuk
import functions as mole
from classes import *
import taichi as ti
import precomp

# Initial conditions and graphics.

@ti.kernel
def init():
    ### Particle/molecule masses in u
    # Hydrogen 2 molecule:
    species_m[0] = 2.015927009
    # Helium 4:
    species_m[1] = 4.002603254
    # Argon:
    species_m[2] = 39.948 
    # Oxygen 2 molecule:
    species_m[3] = 31.998
    
    # Adiabatic indeces for T = 20⁰C
    adiabatic[0] = 1.410
    adiabatic[1] = 1.660
    adiabatic[2] = 1.670
    adiabatic[3] = 1.400

    for i, j in q:
        r = i/N_r
        φ = j/N_φ

        for k in ti.static(range(species)):
            X[i, j][k] = 0.

        
        q[i, j] = [0.8, 0, 0, 2]
        X[i, j][0] = 1
        if φ > 0 and φ < 0.25:
            X[i, j][0] = 1
            q[i, j] = [0.5, 5, 5, 604531]
        elif φ > 0.25 and φ < 0.5:
            X[i, j][1] = 1
            X[i, j][0] = 0
            q[i, j] = [0.9, -4, -4, 548055]
        elif φ > 0.5 and φ < 0.75:
            X[i, j][2] = 1
            X[i, j][0] = 0
            q[i, j] = [0.2, 0, 1, 12203]
        elif φ > 0.75 and φ <= 1:
            X[i, j][3] = 1
            X[i, j][0] = 0
            q[i, j] = [0.5, -5, 0, 38087]


@ti.kernel
def paint(display: int):
    for i, j in pixels:
        z = (ti.Vector([i/res, j/res]) - center) / zoom # Center the vector and scale everything
        if z.norm() * zoom < 0.5:
            r, t = index_grid[i, j] # use precomputed cell indexes
            pixels[i, j] = q[r, t][0]
            if display == 1: pixels[i, j] = q[r, t][0]
            elif display == 2: pixels[i, j] = ti.sqrt(q[r, t][1]**2 + q[r, t][2]**2)
            elif display == 3: pixels[i, j] = q[r, t][3] / 600000
            elif display == 4: pixels[i, j] = U[r, t][3]
            elif display == 5: pixels[i, j] = T[r, t] / 500
            elif display == 6: pixels[i, j] = X[r, t][0] # See Molecular hydrogen
            elif display == 7: pixels[i, j] = X[r, t][1] # see Helium 4
            elif display == 8: pixels[i, j] = X[r, t][2] # see argon
            elif display == 9: pixels[i, j] = X[r, t][3] # see molecular oxygen

        else:
            pixels[i, j] = 0

gui = ti.GUI('grid projection', (res, res))


init()
precomp.assign_grid_indexes(index_grid, center, zoom, res, N_r, N_φ, dφ, fov_rad)
nuk.calc_averages()
euler.prim_to_cons()
t = 0.
previous_max_vel = 0.
indicator = ''
display = 1
global pause 
pause = 0
im = 0
while gui.running:
    if t == 0 and im == 0: 
        paint(display)
        gui.text(f't = {t:.3}', pos=[0.8, 0.95])

        gui.set_image(cmap(pixels.to_numpy()))
        filename = f'init_fractions.png'
        gui.show(filename)
        im += 1

    # Run the sim:
    if not pause: 
        euler.Euler_scheme()

    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            if pause: pause = False
            else: pause = True
        elif e.key == '1':
            show_field = 'Density'
            display = 1
        elif e.key == '2':
            show_field = 'Velocity'
            display = 2
        elif e.key == '3':
            show_field = 'Pressure'
            display = 3
        elif e.key == '4':
            show_field = 'Energy'
            display = 4
        elif e.key == '5':
            show_field = 'Temperature'
            display = 5
        elif e.key == '6':
            show_field = 'molecular H'
            display = 6
        elif e.key == '7': 
            show_field = 'Helium 4'
            display = 7
        elif e.key == '8': 
            show_field = 'Argon'
            display = 8
        elif e.key == '9': 
            show_field = 'molecular O2'
            display = 9

    """ # Uncomment this and remove the if statements down below if you wish to monitor conserved quantities and velocities
    tot_mass, tot_rad_mom, tot_phi_mom, tot_rho_E = mole.check_if_conserved()
    max_vel = mole.get_max_vel()

    gui.triangle(a=[0.63, 0.22], b=[0.63, 0.], c=[1., 0.22], color=12342132)
    gui.triangle(a=[0.63, 0.], b=[1., 0.], c=[1., 0.22], color=12342132)

    gui.triangle(a=[0.79, 0.95], b=[0.9, 0.95], c=[0.9, 0.92], color=0)
    gui.triangle(a=[0.79, 0.95], b=[0.79, 0.92], c=[0.9, 0.92], color=0)
    gui.triangle(a=[0.0, 0.28], b=[0.0, 0.0], c=[0.28, 0.0], color=0)
    if max_vel > previous_max_vel:
        indicator = 'INCREASING'
    else: 
        indicator = 'DECREASING'

    
    gui.text(f'Max Velocity = {max_vel:.5}', pos=[0.01, 0.10])
    gui.text(f'% of c: {(max_vel/c_light)*100:.3}', pos=[0.01, 0.05])
    gui.text(f'{indicator}', pos=[0.02, 0.14])
    gui.text(f't = {t:.3}', pos=[0.8, 0.95])
    previous_max_vel = max_vel
    """

    if t > 6e-5 and im == 1:
        paint(display)
        gui.set_image(cmap(pixels.to_numpy()))
        gui.text(f't = {t:.3}', pos=[0.8, 0.95])

        filename = f'fraction1.png'
        gui.show(filename)
        im += 1
    elif t > 6e-4 and im == 2: 
        precomp.assign_grid_indexes(index_grid, center, zoom=2, res=res, N_r=N_r, N_θ=N_φ, dθ=dφ, fov_rad=fov_rad)

        paint(display)
        gui.text(f't = {t:.3}', pos=[0.8, 0.95])

        gui.set_image(cmap(pixels.to_numpy()))
        filename = f'fractionzoom.png'
        gui.show(filename)
        break
    else: 
        paint(display)
        gui.set_image(cmap(pixels.to_numpy()))
        gui.text(f't = {t:.3}', pos=[0.8, 0.95])
        gui.show()

    t += dt[None]
    print(f't = {t}')