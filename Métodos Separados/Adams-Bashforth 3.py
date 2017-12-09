import numpy as np
import matplotlib.pyplot as plt
import parser
import prettytable
from math import *


class Nummet():
	def __init__(self, x0, y0, xf, n, strfunc):
		self.x0 = x0
		self.y0 = y0
		self.xf = xf
		self.n = n
		self.x = np.linspace(x0, xf, n)
		self.h = (xf-x0)/(n-1) #delta x
		self.deriv = parser.expr(strfunc).compile()
		self.strfunc = strfunc
		
	def adams_bash3(self):
		vetx = np.linspace(self.x0, self.xf, self.n)
		vety = np.zeros(self.n)
		vety [0] = self.y0
		#runge-kutta3 para achar vety[1]e[2]
		for i in range(1, 3):
			k1 = Nummet.f(self, vetx [i-1], vety [i-1])
			k2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*k1)
			k3 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]-k1*self.h+2*k2*self.h)
			vety [i] = vety [i-1]+self.h*(k1+4*k2+k3)/6
		for i in range(3, self.n):
			f1 = Nummet.f(self, vetx [i-1], vety [i-1])
			f2 = Nummet.f(self, vetx [i-2], vety [i-2])
			f3 = Nummet.f(self, vetx [i-3], vety [i-3])
			vety [i] = vety [i-1]+self.h*(23/12*f1-4/3*f2+5/12*f3)
		return vetx, vety
		
	def f(self, x, y):
		return eval(self.deriv)
		
	@staticmethod
	def myplot(a, b, title):
		plt.plot(a, b)
		plt.title(title)
		plt.xlabel('x')
		plt.ylabel('y(x)')
		plt.show()
		
	@staticmethod
	def ptstab(vx, vy):
		tab = prettytable.PrettyTable()
		tab.add_column("x", vx)
		tab.add_column("y Adams-Bashforth 3", vy)
		print(tab)
		
	@classmethod
	def fromstr(cls, strin):
		x0, y0, xf, n, strfunc = strin.split(', ')
		return cls(float(x0), float(y0), float(xf), int(n), strfunc)
		
		
plt.grid()
strin = input('''Digite um problema de valor inicial na forma: "x0, y0, xf, n, y'". Por exemplo: "0, 4, 10, 20, (x-y)/(x+y)"\n''')
vetx, vety = Nummet.adams_bash3(Nummet.fromstr(strin))
Nummet.ptstab(vetx, vety)
Nummet.myplot(vetx, vety, 'Adams-Bashforth 3')
