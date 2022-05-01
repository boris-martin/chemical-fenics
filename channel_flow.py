from fenics import *
from mshr import *
import numpy as np
import fluidutils


# Physical quantities
T = 10.0           # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density

# Mesh
length = 4
p_in = 8
domain = Rectangle(Point(0,0), Point(length, 1)) #- Circle(Point(0.5, 0.5), 0.1)
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Boundary conditions

inflow = 'near(x[0], {})'.format(0)
outflow = 'near(x[0], {})'.format(length)
walls = 'near(x[1], 0) || near(x[1], 1) || (on_boundary && ((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) < 0.02)'

bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(p_in), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

solver = fluidutils.FluidSolver({'mu' : mu, 'rho' : rho}, mesh, bcu, bcp, dt, V, Q)

for n in range(num_steps):

    solver.step()