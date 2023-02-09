# Astro-Gymnasiearbete-programkod

<img src="https://user-images.githubusercontent.com/124574038/217674246-e434f6de-2c13-49f1-baad-15681ec1af7d.png" width=45% height=45%>  <img src="https://user-images.githubusercontent.com/124574038/217681683-101a1fd8-8a32-4f39-92ef-db1be3b7ed73.png" width=45% height=45%>


A simulations program for solving the euler equations of motion for a fluid in 2D-spherical coordinates. 
As of yet I am still developing the source code, adding features to it etc. It's currently bit of a mess, so I'll only upload the code that is completely finished. You'll notice that many of the separate projects/simulations/examples contain the same files. That's just me being a bit lazy with the file structure, managing each part of the code separately. Sometime in the future I'll reorganize everything to look a bit nicer, optimize and improve file structure.

The code is written entirely in Taichi, a language embedded in python that makes it possible to write parallelized code running on the graphics card, yielding much, much faster runtimes for applications that lends themselves well to `for` loops, like multicell simulations. 

This entire project was made (is currently being made) for a school project, so if you just happened to find this page...well I didn't really have external users in mind so apologies if you don't find it user-friendly.

## Current Features:
- 2D-Spherical coordinates
- Compressible Hydrodynamics as described by the [Euler equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics))
- Newtonian simulation
- Simulation compatible with motion as described by special relativity
- Multiple species of molecular/atomic gases, each with its own adiabatic index Γ
- Newtonian, spherically symmetric, gravity
- Custom unit conversion (in case the numbers involved become to great for the computer to handle accurately, will have to edit manually)
- Large scale simulation!

###### To-do
- Fusion!
- Quantum mechanical phenomena (electron degeneracy pressure for example)
- Photon- and ideal gas mixture
- Urca processes
- Temperature diffusion

## Installation and running
###### Dependencies
You will need the following python packages
- Taichi
- matplotlib
- numpy
###### Installation
1. Install the dependencies above first: `pip install taichi matplotlib numpy`
2. Copy the repository: `git clone https://github.com/rasmusduedev/Astro-Gymnasiearbete-programkod.git`
