"""
FEniCS tutorial demo program: Deflection of a membrane.

  -Laplace(w) = p  in the unit circle
            w = 0  on the boundary

The load p is a Gaussian function centered at (0, 0.6).
"""

from __future__ import print_function
import fenics as FnX
import mshr as MsH
import numpy as np

# Create mesh and define function space
domain = MsH.Circle(FnX.Point(0, 0), 1)
mesh = MsH.generate_mesh(domain, 64)
V = FnX.FunctionSpace(mesh, 'P', 2)

# Define boundary condition
w_D = FnX.Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = FnX.DirichletBC(V, w_D, boundary)

# Define load
beta = 8
R0 = 0.6
p = FnX.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
               degree=1, beta=beta, R0=R0)

# Define variational problem
w = FnX.TrialFunction(V)
v = FnX.TestFunction(V)
a = FnX.dot(FnX.grad(w), FnX.grad(v))*FnX.dx
L = p*v*FnX.dx

# Compute solution
w = FnX.Function(V)
FnX.solve(a == L, w, bc)

# Interpolate load 
p = FnX.interpolate(p, V)
# plot(w, title='Deflection')
# plot(p, title='Load')
#NO

# Save solution to file in VTK format
vtkfile_w = FnX.File('poisson_membrane/deflection.pvd')
vtkfile_w << w
vtkfile_p = FnX.File('poisson_membrane/load.pvd')
vtkfile_p << p

# Curve plot along x = 0 comparing p and w
import numpy as np
import matplotlib.pyplot as plt
tol = 0.001  # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y]  # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])
plt.plot(y, 50*w_line, 'k', linewidth=2)  # magnify w
plt.plot(y, p_line, 'b--', linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
plt.savefig('poisson_membrane/curves.pdf')
plt.savefig('poisson_membrane/curves.png')

# Hold plots
# interactive()
plt.show()
