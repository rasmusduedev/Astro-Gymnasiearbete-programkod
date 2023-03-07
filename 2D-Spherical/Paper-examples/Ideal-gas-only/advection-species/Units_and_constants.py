import taichi as ti
ti.init(ti.gpu)

"""
Class:
   🟦 Newtonian
   ✅ Relativistic

Dimensions:
   🟦 1D
   ✅ 2D
   🟦 3D

Coordinate System:
   🟦 Cartesian
   ✅ Spherical

Integration Method:
   🟦 1st Order Euler
   🟦 2nd Order Extrapolate faces in time
   ✅ Rk2
   🟦 Rk4

Boundary Conditions:
   ✅ Reflective both
   🟦 Reflective core, outflow surface

Stable?:
   🟦 Yes
   🟦 No
   ✅ As far as I've tested
   🟦 Sometimes
                                                                
"""

# Grid
N_r = 150 # radius cells
N_φ = 400 # polar angle cells
degree_view = 360 # span of the circle, in degrees [do notice that the angle is with respect to the right horizontal line, going anticlockwise]
res = 800
zoom = 0.5 
cmap_name = 'jet'  # python colormap

# Constants
C = 0.8
species = 4

# In SI-Units
# SI Constants
C = 0.4
SI_c = 299792458 # m/s                       Speed of light in vacuum
SI_k_B = 1.380649*1e-23 # m² kg / (s² K)     Boltzmann's constant
SI_u = 1.66053907*1e-27 # kg                 Atomic mass unit
SI_G = 6.67430 * 1e-11  # N m² / kg²         Gravitational constant
SI_a = 7.5657 * 1e-16   # J / (m^3 K⁴)       Radiation constant
SI_e_charge = 1.60217646 * 1e-19 # Coulomb   Elementary charge
SI_h = 6.62607015 * 1e-34 # m² kg / s        Planck constant

# Define new units for the sim, the units themselves are expressed in SI-units but the simulation code only sees a length_unit as '1' unit length. 
length_unit = 1
mass_unit = 1
time_unit = 1
Current_unit = 1
Temperature_unit = 1
density_unit = mass_unit / (length_unit**3)
energy_unit = mass_unit * length_unit**2 / (time_unit**2)
pressure_unit = energy_unit / (length_unit**3)
velocity_unit = length_unit / time_unit
Force_unit = mass_unit * length_unit / (time_unit**2)
acceleration_unit = length_unit / (time_unit**2)
electric_charge_unit = Current_unit * time_unit

# In order to get the simulation variants of the natural constants we divide each by their corresponding sim-unit:
c_light         = SI_c / length_unit
k_B             = SI_k_B / (length_unit**2 * mass_unit / (time_unit**2 * Temperature_unit))
u_in_mass_units = SI_u / (mass_unit)
G               = SI_G / (Force_unit * length_unit**2 / (mass_unit**2))
a_rad           = SI_a / (energy_unit / (length_unit**3 * Temperature_unit**4))
e_charge        = SI_e_charge / (electric_charge_unit)
h_planck        = SI_h / (length_unit**2 * mass_unit / time_unit)



if __name__ == "__main__":
   print('\n ##### New Units #####')
   print(f'Length = {length_unit} meters\nmass = {mass_unit} kg\ntime = {time_unit} s')
   print(f'Density = {density_unit} kg/m³\nEnergy = {energy_unit} J\nPressure = {pressure_unit} J/m³')
   print(f'Velocity = {velocity_unit} m/s\nForce = {Force_unit} N\nTemperature = {Temperature_unit} K')
   print(f'Current = {Current_unit} A\nElectric Charge = {electric_charge_unit} C')
   
   print(f'\n\n##### New Constants #####')
   print(f'c_light = {c_light} new_units')
   print(f'Boltzmann: {k_B} new_units\nu = {u_in_mass_units} new_units\nG = {G} new_units\na = {a_rad} new_units')
   print(f'Elementary charge e = {e_charge} new_units\nh = {h_planck} new_units')

RADIUS = 1*length_unit