from fenics import *
from mshr import *


domain = Rectangle(Point(0,0), Point(2.2, 0.41)) - Circle(Point(0.2, 0.2), 0.05)
mesh = generate_mesh(domain, 64)
normal = FacetNormal(mesh)

# Manufactured solution : 1 + x² + 2y² => -laplacian = f is -6
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('x[0]*x[0] + 2*x[1]*x[1] + 1', degree=2)

def boundary(x, on_boundary):
    return on_boundary and (not near(x[0], 1) or near(x[1], 0) or near(x[1], 1))

bc = DirichletBC(V, u_D, boundary)

# Formulate the problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
g = Constant(2.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx + g * v * ds

u = Function(V)
u.rename('data', 'Main scalar')

A = assemble(a)
b = assemble(L)
bc.apply(A, b)

solve(A, u.vector(), b)

vtkfile = File('out/flow.pvd')
vtkfile << u