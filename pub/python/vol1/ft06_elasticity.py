"""
FEniCS tutorial demo program: Linear elastic problem.

  -div(sigma(u)) = f

The model is used to simulate an elastic beam clamped at
its left end and deformed under its own weight.
"""

from __future__ import print_function
import fenics as FnX

# Scaled variables
L = 2; W = 0.2
mu = 2
rho = 2
delta = W/L
gamma = 0.8*delta**2
beta = 1.75
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = FnX.BoxMesh(FnX.Point(0, 0, 0), FnX.Point(L, W, W), 10, 3, 3)
V = FnX.VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = FnX.DirichletBC(V, FnX.Constant((0, 0, 0)), clamped_boundary)

# Define strain and stress

def epsilon(u):
    return 0.5*(FnX.nabla_grad(u) + FnX.nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_*FnX.div(u)*FnX.Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = FnX.TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = FnX.TestFunction(V)
f = FnX.Constant((0, 0, -rho*g))
T = FnX.Constant((0, 0, 0))
a = FnX.inner(sigma(u), epsilon(v))*FnX.dx
L = FnX.dot(f, v)*FnX.dx + FnX.dot(T, v)*FnX.ds

# Compute solution
u = FnX.Function(V)
FnX.solve(a == L, u, bc)

# Plot solution
# FnX.plot(u, title='Displacement', mode='displacement')

# Plot stress
s = sigma(u) - (1./3)*FnX.tr(sigma(u))*FnX.Identity(d)  # deviatoric stress
von_Mises = FnX.sqrt(3./2*FnX.inner(s, s))
V = FnX.FunctionSpace(mesh, 'P', 1)
von_Mises = FnX.project(von_Mises, V)
# FnX.plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = FnX.sqrt(FnX.dot(u, u))
u_magnitude = FnX.project(u_magnitude, V)
FnX.plot(u_magnitude, 'Displacement magnitude')
vertex_values_u_magnitude = u_magnitude.compute_vertex_values(mesh)
import numpy as np
print('min/max u:',
     np.min(np.abs(vertex_values_u_magnitude)),
     np.max(np.abs(vertex_values_u_magnitude)))

# Save solution to file in VTK format
FnX.File('elasticity/displacement.pvd') << u
FnX.File('elasticity/von_mises.pvd') << von_Mises
FnX.File('elasticity/magnitude.pvd') << u_magnitude

# Hold plot
import matplotlib.pyplot as plt
plt.show()
