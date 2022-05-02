from fenics import *
from mshr import *

T = 10.0           # final time
num_steps = 100    # number of time steps
dt = T / num_steps # time step size

domain = Rectangle(Point(0,0), Point(4, 1))
mesh = generate_mesh(domain, 32)
normal = FacetNormal(mesh)

# Manufactured solution : 1 + x² + 2y² => -laplacian = f is -6
P = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P, P]))

# No BCs: only homgenuous Neumann


# Create used terms
u_ = Function(V)
u_n = Function(V)

# Formulate the problem

u = TrialFunction(V)
v = TestFunction(V)
f = Expression(('x[0]/4', '1 - x[0]/4'), degree=1)
k = Constant(1. / dt)

u_1, u_2 = split(u)
v_1, v_2 = split(v)
u_n1, u_n2 = split(u_n)
#F = k * dot((u - u_n), v) * dx + inner(nabla_grad(u), nabla_grad(v)) * dx - dot(f, v) * dx
F = k * (u_1 - u_n1) * v_1 * dx + dot(grad(u_1), grad(v_1)) * dx - dot(f[0], v_1) * dx + \
    k * (u_2 - u_n2) * v_2 * dx + dot(grad(u_2), grad(v_2)) * dx - dot(f[1], v_2) * dx
a = lhs(F)
L = rhs(F)

u = Function(V)
u_.rename('data', 'Main scalar')

A = assemble(a)
b = assemble(L)


t = 0
vtkfileA = File('out/chemical_A.pvd')
vtkfileB = File('out/chemical_B.pvd')

for n in range(num_steps):

    t += dt 
    print("Time {} of {}".format(t, T))
    b = assemble(L) # Update usage of u_n
    # A must be re-assembled only with changing time step
    solve(A, u_.vector(), b)

    u_n.assign(u_)

    u_A, u_B = u_.split()
    u_A.rename('data', 'data')
    u_B.rename('data', 'data')
    vtkfileA << u_A, t
    vtkfileB << u_B, t


vtkfile = File('out/poisson.pvd')
vtkfile << u