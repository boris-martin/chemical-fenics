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
p_in = 8*4
domain = Rectangle(Point(0,0), Point(length, 1)) #- Circle(Point(0.5, 0.5), 0.1)
mesh = generate_mesh(domain, 50)

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

fluid_solver = fluidutils.FluidSolver({'mu' : mu, 'rho' : rho}, mesh, bcu, bcp, dt, V, Q)

# Define chemical problem
# Generate chemical component in a circle of radius 0.1 on 0.5, 0.5
f = Expression('pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) <= 0.1*0.1 ? 1.0 : 0.0', degree=2)
diff = Constant(0.1)
k = Constant(1./dt)

c = TrialFunction(Q)
v = TestFunction(Q)
c_n = Function(Q)
c_ = Function(Q)

a_diff = k * (c * v * dx) + diff * dot(grad(c), grad(v)) * dx + dot(fluid_solver.u_, grad(c)) * v * dx
L_diff = k * (c_n * v * dx) + f * v * dx

t = 0
vtk_c = File('out/chemical.pvd')
for n in range(num_steps):

    t += dt 
    fluid_solver.step()
    solve(a_diff == L_diff, c_)
    c_n.assign(c_)
    vtk_c << c_, t