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
		
	def adams_moul5(self):
		vetx = np.linspace(self.x0, self.xf, self.n)
		vety = np.zeros(self.n)
		vety [0] = self.y0
		#runge-kutta5 para achar vety[i] estimado
		for i in range(1, self.n):
			f1 = Nummet.f(self, vetx [i-1], vety [i-1])
			f2 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/4*f1)
			f3 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/8*f1+self.h/8*f2)
			f4 = Nummet.f(self, vetx [i-1]+self.h/2, vety [i-1]-self.h/2*f2+self.h*f3)
			f5 = Nummet.f(self, vetx [i-1]+self.h*3/4, vety [i-1]+self.h*3/16*f1+self.h*9/16*f4)
			f6 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*(-3/7*f1+2/7*f2+12/7*f3-12/7*f4+8/7*f5))
			vety [i] = vety [i-1]+self.h*(7*f1+32*f3+12*f4+32*f5+7*f6)/90
		for i in range(4, self.n):
			k0 = Nummet.f(self, vetx [i], vety [i])
			k1 = Nummet.f(self, vetx [i-1], vety [i-1])
			k2 = Nummet.f(self, vetx [i-2], vety [i-2])
			k3 = Nummet.f(self, vetx [i-3], vety [i-3])
			k4 = Nummet.f(self, vetx [i-4], vety [i-4])
			vety [i] = vety [i-1]+self.h*(251/720*k0+323/360*k1-11/30*k2+53/360*k3-19/720*k4)
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
		tab.add_column("y Adams-Moulton 5", vy)
		print(tab)
		
	@classmethod
	def fromstr(cls, strin):
		x0, y0, xf, n, strfunc = strin.split(', ')
		return cls(float(x0), float(y0), float(xf), int(n), strfunc)
		
		
plt.grid()
strin = input('''Digite um problema de valor inicial na forma: "x0, y0, xf, n, y'". Por exemplo: "0, 4, 10, 20, (x-y)/(x+y)"\n''')
vetx, vety = Nummet.adams_moul5(Nummet.fromstr(strin))
Nummet.ptstab(vetx, vety)
Nummet.myplot(vetx, vety, 'Adams-Moulton 5')

