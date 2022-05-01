from fenics import *

mu = 1.0

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))
# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


class FluidSolver:

    def __init__(self, properties, mesh, bcu, bcp, dt, V, Q):
        self.properties = properties
        self.mesh = mesh
        self.bcu = bcu
        self.bcp = bcp
        self.dt = dt

        self.V = V
        self.Q = Q

        # Symbols for formulating the problem
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        # Define functions for solutions at previous and current time steps
        self.u_n = Function(V)
        self.u_  = Function(V)
        self.u_.rename('Velocity', 'Velocity field')
        self.p_n = Function(Q)
        self.p_  = Function(Q)
        self.p_.rename('Pressure', 'Pressure field')

        # Objects for the variational problem
        f = Constant((0, 0))
        k = Constant(dt)
        mu = Constant(properties['mu'])
        rho = Constant(properties['rho'])
        self.U = 0.5*(self.u_n + u)
        n = FacetNormal(mesh)

        # Define strain-rate tensor and stress tensor
        epsilon = lambda u : sym(nabla_grad(u))
        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(len(u))

        # Step 1: predict velocity with initial pressure
        F1 = rho * dot((u-self.u_n) / k, v) * dx + \
            rho * dot(dot(self.u_n, nabla_grad(self.u_n)), v) * dx + \
            inner(sigma(self.U, self.p_n), epsilon(v)) * dx +\
            dot(self.p_n * n, v) * ds - dot(mu * nabla_grad(self.U) * n, v) * ds - \
            dot(f, v) * dx

        a1 = lhs(F1)
        L1 = rhs(F1)
        A1 = assemble(a1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(self.p_n), nabla_grad(q))*dx - (1/k)*div(self.u_)*q*dx
        A2 = assemble(a2)

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(self.u_, v)*dx - k*dot(nabla_grad(self.p_ - self.p_n), v)*dx
        A3 = assemble(a3)

        # Apply boundary conditions to matrices (why not A3 ?)
        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]

        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # Time-stepping
        self.t = 0
        self.vtkfile_u = File('out/velocity.pvd')
        self.vtkfile_p = File('out/pressure.pvd')

        pass

    def step(self):
        # Update current time
        self.t += self.dt
        print("Time is {}".format(self.t))

        # Step 1: Tentative velocity step
        b1 = assemble(self.L1)
        [bc.apply(b1) for bc in self.bcu]
        solve(self.A1, self.u_.vector(), b1)

        # Step 2: Pressure correction step
        b2 = assemble(self.L2)
        [bc.apply(b2) for bc in self.bcp]
        solve(self.A2, self.p_.vector(), b2)

        # Step 3: Velocity correction step
        b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), b3)

        # Update previous solution
        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)

        self.vtkfile_u << self.u_, self.t
        self.vtkfile_p << self.p_, self.t