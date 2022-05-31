from fenics import *
from mshr import *

T = 1.0           # final time
num_steps = 10    # number of time steps
dt = T / num_steps # time step size

domain = Rectangle(Point(0,0), Point(4, 1))
mesh = generate_mesh(domain, 32)
normal = FacetNormal(mesh)

# Three dimensional vector for three species
P = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P, P, P]))

# No BCs: only homgenuous Von Neumann. It means nothing leaves the domain
# Von neumann BCs give rise to surface loads, which are simply zero here.


# Create used terms (_n suffix means "at previous step")
u_ = Function(V)
u_n = Function(V)
v = TestFunction(V)
u_1, u_2, u_3 = split(u_)
v_1, v_2, v_3 = split(v)

# Formulate the problem

f = Expression(('x[0]/4', '1 - x[0]/4'), degree=1)
k = Constant(1. / dt)
r = Constant(1.0) # Reaction speed
diff = Constant(0.01) # Diffusivity

# Velocity field (from preCICE in the future)
flow = Expression(("vmax * 4*x[1]*(1-x[1])", "0"), degree=2, vmax=0.5)

u_n1, u_n2, u_n3 = split(u_n)
F = k * (u_1 - u_n1) * v_1 * dx + 0.5 * dot(flow, grad(u_1 + u_n1)) * v_1 * dx + diff * dot(grad(u_1), grad(v_1)) * dx - dot(f[0], v_1) * dx + r * u_1 * u_2 * v_1 * dx + \
    k * (u_2 - u_n2) * v_2 * dx + 0.5 * dot(flow, grad(u_2 + u_n2)) * v_2 * dx + diff * dot(grad(u_2), grad(v_2)) * dx - dot(f[1], v_2) * dx + r * u_1 * u_2 * v_2 * dx + \
    k * (u_3 - u_n3) * v_3 * dx + 0.5 * dot(flow, grad(u_3 + u_n3)) * v_3 * dx + diff * dot(grad(u_3), grad(v_3)) * dx                       - r * u_1 * u_2 * v_3 * dx



t = 0
vtkfileA = File('out/chemical_A.pvd')
vtkfileB = File('out/chemical_B.pvd')
vtkfileC = File('out/chemical_C.pvd')

for n in range(num_steps):

    t += dt 
    print("Time {:.3g} of {:.3g}".format(t, T))
    solve(F == 0, u_)
    u_n.assign(u_)

    u_A, u_B, u_C = u_.split()
    u_A.rename('data', 'data')
    u_B.rename('data', 'data')
    u_C.rename('data', 'data')

    vtkfileA << u_A, t
    vtkfileB << u_B, t
    vtkfileC << u_C, t
