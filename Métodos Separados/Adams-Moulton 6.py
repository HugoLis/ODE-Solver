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
		
	def adams_moul6(self):
		vetx = np.linspace(self.x0, self.xf, self.n)
		vety = np.zeros(self.n)
		vety [0] = self.y0
		#runge-kutta6 para achar vety[i] estimado
		for i in range(1, self.n):
			f1 = Nummet.f(self, vetx [i-1], vety [i-1])
			f2 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+f1*self.h)
			f3 = Nummet.f(self, vetx [i-1]+self.h/2, vety [i-1]+((3*f1+f2)/8)*self.h)
			f4 = Nummet.f(self, vetx [i-1]+self.h*2/3, vety [i-1]+((8*f1+2*f2+8*f3)/27)*self.h)
			f5 = Nummet.f(self, vetx [i-1]+self.h*(7-21**(1/2))/14, vety [i-1]+(
			((3*(3*21**(1/2)-7))*f1-(8*(7-21**(1/2)))*f2+(48*(7-21**(1/2)))*f3-(3*(21-21**(1/2)))*f4)/392)*self.h)
			f6 = Nummet.f(self, vetx [i-1]+self.h*(7+21**(1/2))/14, vety [i-1]+((-(5*(231+51*21**(1/2)))*f1-(
			40*(7+21**(1/2)))*f2-(320*21**(1/2))*f3+(3*(21+121*21**(1/2)))*f4+(392*(6+21**(1/2)))*f5)/1960)*self.h)
			f7 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+(((15*(22+7*21**(1/2)))*f1+(120)*f2+(
			40*(7*21**(1/2)-5))*f3-(63*(3*21**(1/2)-2))*f4-(14*(49+9*21**(1/2)))*f5+(
			70*(7-21**(1/2)))*f6)/180)*self.h)
			vety [i] = vety [i-1]+self.h*(9*f1+64*f3+49*f5+49*f6+9*f7)/180
		for i in range(5, self.n):
			k0 = Nummet.f(self, vetx [i], vety [i])
			k1 = Nummet.f(self, vetx [i-1], vety [i-1])
			k2 = Nummet.f(self, vetx [i-2], vety [i-2])
			k3 = Nummet.f(self, vetx [i-3], vety [i-3])
			k4 = Nummet.f(self, vetx [i-4], vety [i-4])
			k5 = Nummet.f(self, vetx [i-5], vety [i-5])
			vety [i] = vety [i-1]+self.h*(95/288*k0+1427/1440*k1-133/240*k2+241/720*k3-173/1440*k4+3/160*k5)
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
		tab.add_column("y Adams-Moulton 6", vy)
		print(tab)
		
	@classmethod
	def fromstr(cls, strin):
		x0, y0, xf, n, strfunc = strin.split(', ')
		return cls(float(x0), float(y0), float(xf), int(n), strfunc)
		
		
plt.grid()
strin = input('''Digite um problema de valor inicial na forma: "x0, y0, xf, n, y'". Por exemplo: "0, 4, 10, 20, (x-y)/(x+y)"\n''')
vetx, vety = Nummet.adams_moul6(Nummet.fromstr(strin))
Nummet.ptstab(vetx, vety)
Nummet.myplot(vetx, vety, 'Adams-Moulton 6')
