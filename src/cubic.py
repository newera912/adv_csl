__author__ = 'fengchen'
import math
import numpy
from math import *
# for i in range(1000):
#     mu = i/1000.0
#     print mu, 3 * math.sqrt(1 - mu * mu) * (1 + 2 * mu)

"""
cbrt(x) = x^{1/3},  if x >= 0
        = -|x|^{1/3},  if x < 0
"""
def cbrt(x):
    from math import pow
    if x >= 0:
	return pow(x, 1.0/3.0)
    else:
	return -pow(abs(x), 1.0/3.0)

def quadratic(a, b, c=None):
    import math, cmath
    if c:		# (ax^2 + bx + c = 0)
	a, b = b / float(a), c / float(a)
    t = a / 2.0
    r = t**2 - b
    if r >= 0:		# real roots
	y1 = math.sqrt(r)
    else:		# complex roots
	y1 = cmath.sqrt(r)
    y2 = -y1
    return y1 - t, y2 - t

"""
Convert from rectangular (x,y) to polar (r,w)
    r = sqrt(x^2 + y^2)
    w = arctan(y/x) = [-\pi,\pi] = [-180,180]
"""
def polar(x, y, deg=0):		# radian if deg=0; degree if deg=1
    from math import hypot, atan2, pi
    if deg:
	return hypot(x, y), 180.0 * atan2(y, x) / pi
    else:
	return hypot(x, y), atan2(y, x)

#real, number, number = cubic(real, real, real [, real])
"""
x^3 + ax^2 + bx + c = 0 (or ax^3 + bx^2 + cx + d = 0)
With substitution x = y-t and t = a/3, the cubic equation reduces to
y^3 + py + q = 0,
where p = b-3t^2 and q = c-bt+2t^3. Then, one real root y1 = u+v can
be determined by solving
w^2 + qw - (p/3)^3 = 0
where w = u^3, v^3. From Vieta's theorem,
y1 + y2 + y3 = 0
y1 y2 + y1 y3 + y2 y3 = p
y1 y2 y3 = -q,
the other two (real or complex) roots can be obtained by solving
y^2 + (y1)y + (p+y1^2) = 0
"""
def cubic(a, b, c, d=None):
    from math import cos
    if d:			# (ax^3 + bx^2 + cx + d = 0)
	a, b, c = b / float(a), c / float(a), d / float(a)
    t = a / 3.0
    p, q = b - 3 * t**2, c - b * t + 2 * t**3
    u, v = quadratic(q, -(p/3.0)**3)
    if type(u) == type(0j):	# complex cubic root
	r, w = polar(u.real, u.imag)
	y1 = 2 * cbrt(r) * cos(w / 3.0)
    else:			# real root
        y1 = cbrt(u) + cbrt(v)
    y2, y3 = quadratic(y1, p + y1**2)
    return y1 - t, y2 - t, y3 - t


# import sympy
import numpy as np
#import matplotlib.pyplot as plt
# x = sympy.Symbol('x')
# sol =  sympy.solve('x^3 - 2*x^2 + x + 11', x)
# print sol[0]
# a = cubic(1, -2, 1, 11)
# print a
# print type(a[0]), a[0]
# if type(a[1]) is complex:
#     print 'complex'


# x = 1.2327856159383832+0.7925519925154465j
# print x * x * x - 2*x * x + x + 11
#
# x = np.arange(-5., 5., 0.1)
#plt.plot(x, x**3 - 2*x**2 + x + 11, 'r--')
#plt.grid()
#plt.show()

# print 2/3 - 1/(3*(-1/2 - sqrt(3)*I/2)*(3*sqrt(93)/2 + 29/2)**(1/3)) - (-1/2 - sqrt(3)*I/2)*(3*sqrt(93)/2 + 29/2)**(1/3)/3
