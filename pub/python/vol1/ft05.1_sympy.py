from fenics import *
import sympy as sym
def q(u):
    return 1 + u**2
x,y=sym.symbols('x[0],x[1]')
u=1+x+2*y
print(u)
f=-sym.diff(q(u)*sym.diff(u,x),x) - sym.diff(q(u)*sym.diff(u,y),y)
print(f)
