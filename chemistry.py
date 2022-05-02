from fenics import *
from mshr import *

T = 10.0           # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size

domain = Rectangle(Point(0,0), Point(4, 1))
mesh = generate_mesh(domain, 32)
normal = FacetNormal(mesh)

# Manufactured solution : 1 + x² + 2y² => -laplacian = f is -6
V = FunctionSpace(mesh, 'P', 1)

# No BCs: only homgenuous Neumann


# Create used terms
u_ = Function(V)
u_n = Function(V)

# Formulate the problem

u = TrialFunction(V)
v = TestFunction(V)
#f = Expression(('x/4', '1 - x/4'))
f = Expression('x[0]/4', degree=1)
k = Constant(1. / dt)
F = k * (u - u_n) * v * dx + dot(grad(u), grad(v)) * dx - f * v * dx
a = lhs(F)
L = rhs(F)

u = Function(V)
u.rename('data', 'Main scalar')

A = assemble(a)
b = assemble(L)


t = 0
vtkfile = File('out/chemical_pure.pvd')
for n in range(num_steps):

    t += dt 
    print("Time {} of {}".format(t, T))
    b = assemble(L) # Update usage of u_n
    solve(A, u_.vector(), b)

    u_n.assign(u_)
    vtkfile << u_, t

vtkfile = File('out/poisson.pvd')
vtkfile << u