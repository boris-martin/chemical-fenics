from termios import VEOL
from fenics import *
from mshr import *
import fenicsprecice
import numpy as np

T = 10.0           # final time
num_steps = 100    # number of time steps
default_dt = T / num_steps # time step size

domain = Rectangle(Point(0,0), Point(4, 1))
mesh = generate_mesh(domain, 32)
normal = FacetNormal(mesh)

# Three dimensional vector for three species
P = FiniteElement('P', triangle, 1)
W = FunctionSpace(mesh, MixedElement([P, P])) # Velocity field is 2D

# No BCs: only homgenuous Von Neumann. It means nothing leaves the domain
# Von neumann BCs give rise to surface loads, which are simply zero here.

class CouplingDomain(SubDomain):
    def inside(self, x, on_boundary):
        return True

# Initialize preCICE
flow = Expression(("vmax * 4*x[1]*(1-x[1])", "0"), degree=2, vmax=0.5)
velocity = interpolate(flow, W)
precice = fenicsprecice.Adapter(adapter_config_filename="flow-config.json")
precice_dt = precice.initialize(coupling_subdomain=CouplingDomain(), write_object=velocity)


dt = np.min([default_dt, precice_dt])

# No implicit coupling
t = 0
while precice.is_coupling_ongoing():

    velocity = interpolate(flow, W)
    precice.write_data(velocity)
    # If we add writing, do it here

    dt = np.min([default_dt, precice_dt])

    t += dt 

    precice_dt = precice.advance(dt)
