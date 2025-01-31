"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
import fenics as FnX
import numpy as np
import matplotlib.pyplot as plt

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = FnX.UnitSquareMesh(nx, ny)
V = FnX.FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = FnX.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = FnX.DirichletBC(V, u_D, boundary)

# Define initial value
u_n = FnX.interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = FnX.TrialFunction(V)
v = FnX.TestFunction(V)
f = FnX.Constant(beta - 2 - 2*alpha)

F = u*v*FnX.dx + dt*FnX.dot(FnX.grad(u), FnX.grad(v))*FnX.dx - (u_n + dt*f)*v*FnX.dx
a, L = FnX.lhs(F), FnX.rhs(F)

# Time-stepping
u = FnX.Function(V)
t = 0
import numpy as np
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    FnX.solve(a == L, u, bc)

    # Plot solution
    FnX.plot(u)

    # Compute error at vertices
    u_e = FnX.interpolate(u_D, V)
    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_= np.max(np.abs(vertex_values_u_e - vertex_values_u))
    print('t = %.2f: error = %.3g' % (t, error_))

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.show()
