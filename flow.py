from fenics import *
from matplotlib import interactive
from sympy import degree


mesh = UnitSquareMesh(8, 8)

# Manufactured solution : 1 + x² + 2y² => -laplacian = f is -6
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('x[0]*x[0] + 2*x[1]*x[1] + 1', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Formulate the problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

u = Function(V)
solve(a == L, u, bc)

vtkfile = File('out/poisson.pvd')
vtkfile << u