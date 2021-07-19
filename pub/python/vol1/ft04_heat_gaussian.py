"""
FEniCS tutorial demo program: Diffusion of a Gaussian hill.

  u'= Laplace(u) + f  in a square domain
  u = u_D             on the boundary
  u = u_0             at t = 0

  u_D = f = 0

The initial condition u_0 is chosen as a Gaussian hill.
"""

from __future__ import print_function
import fenics as FnX
import time

T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 30
mesh = FnX.RectangleMesh(FnX.Point(-2, -2), 
                         FnX.Point(2, 2), nx, ny)
V = FnX.FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = FnX.DirichletBC(V, FnX.Constant(0), boundary)

# Define initial value
u_0 = FnX.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
                 degree=2, a=5)
u_n = FnX.interpolate(u_0, V)

# Define variational problem
u = FnX.TrialFunction(V)
v = FnX.TestFunction(V)
f = FnX.Constant(0)

F = u*v*FnX.dx + dt*FnX.dot(FnX.grad(u), FnX.grad(v))*FnX.dx - (u_n + dt*f)*v*FnX.dx
a, L = FnX.lhs(F), FnX.rhs(F)

# Create VTK file for saving solution
vtkfile = FnX.File('heat_gaussian/solution.pvd')

# Time-stepping
u = FnX.Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    FnX.solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)
    FnX.plot(u)

    # Update previous solution
    u_n.assign(u)

# Hold plot
from matplotlib.pyplot import show
show()
