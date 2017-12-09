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
		
	def euler(self):
		vetx = np.linspace(self.x0, self.xf, self.n)
		vety = np.zeros(self.n)
		vety [0] = self.y0
		for i in range(1, self.n):
			vety [i] = vety [i-1]+self.h*Nummet.f(self, vetx [i-1], vety [i-1])
		# print ('x = '+str(vetx[i])+' y = '+str(vety[i]))
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
		tab.add_column("y Euler (AB 1 e RK 1)", vy)
		print(tab)
		
	@classmethod
	def fromstr(cls, strin):
		x0, y0, xf, n, strfunc = strin.split(', ')
		return cls(float(x0), float(y0), float(xf), int(n), strfunc)
		
		
plt.grid()
strin = input('''Digite um problema de valor inicial na forma: "x0, y0, xf, n, y'". Por exemplo: "0, 4, 10, 20, (x-y)/(x+y)"\n''')
vetx, vety = Nummet.euler(Nummet.fromstr(strin))
Nummet.ptstab(vetx, vety)
Nummet.myplot(vetx, vety, 'Euler (AB 1 e RK 1)')

