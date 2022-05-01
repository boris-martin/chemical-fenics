from fenics import *
from mshr import *
import numpy as np

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))
# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Physical quantities
T = 10.0           # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density

# Mesh
length = 4
p_in = 8
domain = Rectangle(Point(0,0), Point(length, 1)) - Circle(Point(0.5, 0.5), 0.1)
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
u_.rename('Velocity', 'Velocity field')
p_n = Function(Q)
p_  = Function(Q)
p_.rename('Pressure', 'Pressure field')


U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

inflow = 'near(x[0], {})'.format(0)
outflow = 'near(x[0], {})'.format(length)
walls = 'near(x[1], 0) || near(x[1], 1) || (on_boundary && ((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) < 0.02)'

bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(p_in), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# k is the timestep

# Step 1: predic velocity with initial pressure
F1 = rho * dot((u-u_n) / k, v) * dx + \
     rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx + \
     inner(sigma(U, p_n), epsilon(v)) * dx +\
     dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds - \
     dot(f, v) * dx

a1 = lhs(F1)
L1 = rhs(F1)
A1 = assemble(a1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
A2 = assemble(a2)

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
A3 = assemble(a3)

# Apply boundary conditions to matrices (why not A3 ?)
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Time-stepping
t = 0
vtkfile_u = File('out/velocity.pvd')
vtkfile_p = File('out/pressure.pvd')


for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Plot solution
    plot(u_)

    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().get_local().max())

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    vtkfile_u << u_, t
    vtkfile_p << p_, t