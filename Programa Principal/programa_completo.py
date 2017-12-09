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

    def inv_euler(self): #Moulton 1
        #Usando o Runge-Kutta 1 (Euler) para aproximar vety[y]
        ex, ey = Nummet.euler(self)
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        for i in range(1, self.n):
            vety [i] = vety [i-1]+self.h*Nummet.f(self, vetx [i], ey [i])
        return vetx, vety

    def apr_euler1(self): #Moulton 2
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #Usando o Runge-Kutta 2 para aproximar vety[i]
        vetx, vety = Nummet.apr_euler2(self)
        for i in range(1, self.n):
            f1 = self.h*Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = self.h*Nummet.f(self, vetx [i], vety [i])
            vety [i] = vety [i-1]+(f1+f2)/2
        return vetx, vety

    def apr_euler2(self): #runge-kutta 2
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        for i in range(1, self.n):
            f1 = self.h*Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = self.h*Nummet.f(self, vetx [i], vety [i-1]+f1)
            vety [i] = vety [i-1]+(f1+f2)/2
        return vetx, vety

    def rungekutta3(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        for i in range(1, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]-f1*self.h+2*f2*self.h)
            vety [i] = vety [i-1]+self.h*(f1+4*f2+f3)/6
        return vetx, vety

    def rungekutta4(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        for i in range(1, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f2)
            f4 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*f3)
            vety [i] = vety [i-1]+self.h*(f1+2*f2+2*f3+f4)/6
        return vetx, vety

    def rungekutta5(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        for i in range(1, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/4*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/8*f1+self.h/8*f2)
            f4 = Nummet.f(self, vetx [i-1]+self.h/2, vety [i-1]-self.h/2*f2+self.h*f3)
            f5 = Nummet.f(self, vetx [i-1]+self.h*3/4, vety [i-1]+self.h*3/16*f1+self.h*9/16*f4)
            f6 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*(-3/7*f1+2/7*f2+12/7*f3-12/7*f4+8/7*f5))
            vety [i] = vety [i-1]+self.h*(7*f1+32*f3+12*f4+32*f5+7*f6)/90
        return vetx, vety

    def rungekutta6(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
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
        return vetx, vety

    def adams_bash2(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta2 (euler aprimorado) para achar vety[1]
        a1 = self.h*Nummet.f(self, vetx [0], vety [0])
        a2 = self.h*Nummet.f(self, vetx [1], vety [0]+a1)
        vety [1] = vety [0]+(a1+a2)/2
        for i in range(2, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-2], vety [i-2])
            vety [i] = vety [i-1]+self.h*(3/2*f1-1/2*f2)
        return vetx, vety

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

    def adams_bash4(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta4 para achar vety[1],[2]e[3]
        for i in range(1, 4):
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*k1)
            k3 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*k2)
            k4 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*k3)
            vety [i] = vety [i-1]+self.h*(k1+2*k2+2*k3+k4)/6
        for i in range(4, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-2], vety [i-2])
            f3 = Nummet.f(self, vetx [i-3], vety [i-3])
            f4 = Nummet.f(self, vetx [i-4], vety [i-4])
            vety [i] = vety [i-1]+self.h*(55/24*f1-59/24*f2+37/24*f3-3/8*f4)
        return vetx, vety

    def adams_bash5(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta5 para achar vety[1],[2],[3]e[4]
        for i in range(1, 5):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/4*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h/4, vety [i-1]+self.h/8*f1+self.h/8*f2)
            f4 = Nummet.f(self, vetx [i-1]+self.h/2, vety [i-1]-self.h/2*f2+self.h*f3)
            f5 = Nummet.f(self, vetx [i-1]+self.h*3/4, vety [i-1]+self.h*3/16*f1+self.h*9/16*f4)
            f6 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*(-3/7*f1+2/7*f2+12/7*f3-12/7*f4+8/7*f5))
            vety [i] = vety [i-1]+self.h*(7*f1+32*f3+12*f4+32*f5+7*f6)/90
        for i in range(5, self.n):
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            vety [i] = vety [i-1]+self.h*(1901/720*k1-1387/360*k2+109/30*k3-637/360*k4+251/720*k5)
        return vetx, vety

    def adams_bash6(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta6 para achar vety[1],[2],[3],[4]e[5]
        for i in range(1, 6):
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
        for i in range(6, self.n):
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            k6 = Nummet.f(self, vetx [i-6], vety [i-6])
            vety [i] = vety [i-1]+self.h*(4277/1440*k1-2641/480*k2+4991/720*k3-3649/720*k4+959/480*k5-95/288*k6)
        return vetx, vety

    def adams_bash7(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta6 para achar vety[1],[2],[3],[4],[5]e[6]
        for i in range(1, 7):
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
        for i in range(7, self.n):
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            k6 = Nummet.f(self, vetx [i-6], vety [i-6])
            k7 = Nummet.f(self, vetx [i-7], vety [i-7])
            vety [i] = vety [i-1]+self.h*(
                198721/60480*k1-18637/2520*k2+235183/20160*k3-10754/945*k4+135713/20160*k5-5603/2520*k6+19087/60480*k7)
        return vetx, vety

    def adams_bash8(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta6 para achar vety[1],[2],[3],[4],[5],[6]e[7]
        for i in range(1, 8):
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
        for i in range(8, self.n):
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            k6 = Nummet.f(self, vetx [i-6], vety [i-6])
            k7 = Nummet.f(self, vetx [i-7], vety [i-7])
            k8 = Nummet.f(self, vetx [i-8], vety [i-8])
            vety [i] = vety [i-1]+self.h*(
                16083/4480*k1-1152169/120960*k2+242653/13440*k3-296053/13440*k4+2102243/120960*k5-115747/13440*k6+32863/13440*k7-5257/17280*k8)
        return vetx, vety

    def adams_moul3(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        # runge-kutta3 para achar vety[i] estimado
        for i in range(1, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]-f1*self.h+2*f2*self.h)
            vety [i] = vety [i-1]+self.h*(f1+4*f2+f3)/6
        for i in range(2, self.n):
            k0 = Nummet.f(self, vetx [i], vety [i])
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            vety [i] = vety [i-1]+self.h*(5/12*k0+2/3*k1-1/12*k2)
        return vetx, vety

    def adams_moul4(self):
        vetx = np.linspace(self.x0, self.xf, self.n)
        vety = np.zeros(self.n)
        vety [0] = self.y0
        #runge-kutta4 para achar vety[i] estimado
        for i in range(1, self.n):
            f1 = Nummet.f(self, vetx [i-1], vety [i-1])
            f2 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f1)
            f3 = Nummet.f(self, vetx [i-1]+self.h*0.5, vety [i-1]+self.h*0.5*f2)
            f4 = Nummet.f(self, vetx [i-1]+self.h, vety [i-1]+self.h*f3)
            vety [i] = vety [i-1]+self.h*(f1+2*f2+2*f3+f4)/6
        for i in range(3, self.n):
            k0 = Nummet.f(self, vetx [i], vety [i])
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            vety [i] = vety [i-1]+self.h*(3/8*k0+19/24*k1-5/24*k2+1/24*k3)
        return vetx, vety

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

    def adams_moul7(self):
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
        for i in range(6, self.n):
            k0 = Nummet.f(self, vetx [i], vety [i])
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            k6 = Nummet.f(self, vetx [i-6], vety [i-6])
            vety [i] = vety [i-1]+self.h*(
                19087/60480*k0+2713/2520*k1-15487/20160*k2+586/945*k3-6737/20160*k4+263/2520*k5-863/60480*k6)
        return vetx, vety

    def adams_moul8(self):
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
        for i in range(7, self.n):
            k0 = Nummet.f(self, vetx [i], vety [i])
            k1 = Nummet.f(self, vetx [i-1], vety [i-1])
            k2 = Nummet.f(self, vetx [i-2], vety [i-2])
            k3 = Nummet.f(self, vetx [i-3], vety [i-3])
            k4 = Nummet.f(self, vetx [i-4], vety [i-4])
            k5 = Nummet.f(self, vetx [i-5], vety [i-5])
            k6 = Nummet.f(self, vetx [i-6], vety [i-6])
            k7 = Nummet.f(self, vetx [i-7], vety [i-7])
            vety [i] = vety [i-1]+self.h*(
                5257/17280*k0+139849/120960*k1-4511/4480*k2+123133/120960*k3-88547/120960*k4+1537/4480*k5-11351/120960*k6+275/24192*k7)
        return vetx, vety

    def f(self, x, y):
        return eval(self.deriv)

    def veterro(self, exact, m): #m stands for the vmet index
        exact = parser.expr(exact).compile()
        vx, vy = Nummet.vmet [m](self)
        dif = []
        for i in range(1, self.n): #we star at 1 because vy[0] is already exact
            x = vx [i]
            dif.append(abs(eval(exact)-vy [i]))
        return dif

    def multerror(self, exact, list):
        #list = Nummet.inlist()
        medias = []
        stddevs = []
        for i in range(0, len(list)):
            err = Nummet.veterro(self, exact, list [i])
            medias.append(np.mean((err)))
            stddevs.append(np.std((err)))
        return medias, stddevs, list

    def multplot(self):
        list = Nummet.inlist()
        plt.grid()
        for i in list:
            vx, vy = Nummet.vmet [i](data)
            plt.plot(vx, vy, label=Nummet.vnames [i])
        plt.legend(loc=4)
        plt.show()

    def multplotplus(self, list):
        #list = Nummet.inlist()
        plt.grid()
        vys = []
        for i in list:
            vx, vy = Nummet.vmet [i](data)
            vys.append(vy)
            plt.plot(vx, vy, label=Nummet.vnames [i])
        plt.legend(loc=4)
        plt.show()
        vx = np.linspace(self.x0, self.xf, self.n)
        return vx, vys, list

    def multmetnoplot(self, list):
        #list = Nummet.inlist()
        #plt.grid()
        vys = []
        for i in list:
            vx, vy = Nummet.vmet [i](data)
            vys.append(vy)
        #plt.plot(vx, vy, label=Nummet.vnames [i])
        #plt.legend(loc=4)
        #plt.show()
        vx = np.linspace(self.x0, self.xf, self.n)
        return vx, vys, list

    def multplotex(self, exact, list):
        #list = Nummet.inlist()
        for i in list:
            vx, vy = Nummet.vmet [i](data)
            plt.plot(vx, vy, label=Nummet.vnames [i])

        ex = np.linspace(self.x0, self.xf, self.n)
        ey = []
        for x in ex:
            ey.append(eval(exact))
        plt.plot(ex, ey, label='Exact')
        plt.legend(loc=4)
        plt.show()

    @staticmethod
    def errtable(medias, stddevs, list):
        tab = prettytable.PrettyTable()
        tab.field_names = ['Método', 'Média do Erro', 'Desvio Padrão do Erro']
        for i in range(0, len(list)):
            tab.add_row([Nummet.vnames [list [i]], medias [i], stddevs [i]])
        print(tab)

    @staticmethod
    def ptstable(vx, vys, list):
        tab = prettytable.PrettyTable()
        tab.add_column("x", vx)
        a = 0
        for i in range(0, len(list)):
            tab.add_column('y '+Nummet.vnames [list [i]], vys [i])
        print(tab)

    @staticmethod
    def ptstableplus(vx, vys, list, exact):
        exact = parser.expr(exact).compile()
        yexact = []
        tab = prettytable.PrettyTable()
        tab.add_column('x', vx)
        for i in range(0, len(list)):
            tab.add_column('y '+Nummet.vnames [list [i]], vys [i])
        for x in vx:
            yexact.append(eval(exact))
        tab.add_column('y exato', yexact)
        print(tab)

    @staticmethod
    def medestd(verro):
        return np.mean(verro, axis=0), np.std(verro, axis=0)

    @staticmethod
    def myplot(a, b, title):
        plt.plot(a, b)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

    @staticmethod
    def metstab():
        mtab = prettytable.PrettyTable()
        mtab.field_names = ['Índice', 'Método']
        for i in range(0, len(Nummet.vnames)):
            mtab.add_row([i, Nummet.vnames [i]])
        print(mtab)

    @classmethod
    def fromstr(cls, strin):
        x0, y0, xf, n, strfunc = strin.split(', ')
        return cls(float(x0), float(y0), float(xf), int(n), strfunc)

    @classmethod
    def autoinlist(cls, inp):
        mlist = [int(x) for x in input(inp).split()]
        return mlist

    @classmethod
    def inlist(cls):
        mlist = [int(x) for x in input().split()]
        return mlist

    vmet = [euler, adams_bash2, adams_bash3, adams_bash4, adams_bash5,\
            adams_bash6, adams_bash7, adams_bash8, inv_euler, apr_euler1,\
            adams_moul3, adams_moul4, adams_moul5, adams_moul6, adams_moul7,\
            adams_moul8, apr_euler2, rungekutta3, rungekutta4, rungekutta5, rungekutta6]

    vnames = ['Euler (RK 1 ou AB 1)', 'Adams-Bashforth 2', 'Adams-Bashforth 3', 'Adams-Bashforth 4',
              'Adams-Bashforth 5',\
              'Adams-Bashforth 6', 'Adams-Bashforth 7', 'Adams-Bashforth 8', 'Euler Inverso (AM 1)',\
              'Euler Aprimorado (AM 2)', 'Adams-Moulton 3', 'Adams-Moulton 4', 'Adams-Moulton 5',\
              'Adams-Moulton 6', 'Adams-Moulton 7', 'Adams-Moulton 8', 'Euler Aprimorado 2 (RK 2)',\
              'Runge-Kutta 3', 'Runge-Kutta 4', 'Runge-Kutta 5', 'Runge-Kutta 6']


#strin = '0, 0, 10, 40, sin(x)'
#strin = "0, 1, 1, 11, 2*(y**2+1)/(x**2+4)"
#Nummet.myplot(*Nummet.euler(Nummet.fromstr(strin)), 'Euler')
#str = '0 1 3 4 9 12'
#Nummet.autoinlist(str)


#data = Nummet(0, 1, 1, 10, 'e**x')
#data = Nummet(0, 1, 1, 30, '1-x+4*y')
#data = Nummet(0, 0, 10, 30, 'cos(x)')
#data = Nummet(0, 4, 10, 20, '(x-y)/(x+y)')
#data = Nummet(0, 1, 5, 20, 'cos(x)*y')
#data = Nummet(0, 1, 1, 20, '2*(y**2+1)/(x**2+4)')

#myerr = Nummet.veterro(data, 'e**x', 0)
#Nummet.medestd(myerr)

#a, b, c = Nummet.multploteerror(data, '2**(1/2)*(x**2+8)**(1/2)-x')
#Nummet.errtable(a, b, c)

#a, b, c = Nummet.multerror(data, '2**(1/2)*(x**2+8)**(1/2)-x')
#Nummet.errtable(a, b, c)
#Nummet.ptstableplus(a, b, c, '2**(1/2)*(x**2+8)**(1/2)-x')

#x = np.linspace(data.x0, data.xf, data.n)

#plt.plot(x, e**x, label = 'Exact')
#plt.plot(x, (4*x+19*e**(4*x)-3)/16, label = 'Exact')
#plt.plot(x, np.sin(x), label = 'Exact')
#plt.plot(x, 2**(1/2)*(x**2+8)**(1/2)-x, label = 'Exact')
#plt.plot(x, e**np.sin(x), label = 'Exact')
#plt.plot(x, np.tan(np.arctan(x/2)+pi/4), label = 'Exact')

#plt.legend(loc=4)
#plt.show()

loop = 1
cont = 1
while cont==1:
    while loop==1:
        plt.close()
        opcao = input(
            "Digite o número referente à opção desejada:\n\n0 - Plotar Gráficos\n1 - Analisar Erros\n2 - Plotar Gráfico e Analisar Erros\n3 - Sair do Programa\n\n")
        if opcao=='3':
            cont = 0
            break
        elif (opcao!='2' and opcao!='1' and opcao!='0'):
            cont = 1
            break
        datastr = input(
            '''Digite um problema de valor inicial na forma: x0, y0, xf, n, y'. Por exemplo: "0, 1, 3, 20, cos(x)*y"\n''')
        data = Nummet.fromstr(datastr)
        #data = Nummet(0, 1, 3, 20, 'cos(x)*y')
        if opcao=='0':#plots e pontos
            Nummet.metstab()
            print('Digite os índices dos métodos a serem plotados na forma: "m0 m1 m2". Por exemplo: "0 1 8 17"\n')
            mets = Nummet.inlist()
            a, b, c = Nummet.multmetnoplot(data, mets)
            Nummet.ptstable(a, b, c)
            Nummet.multplotplus(data, mets)
        elif opcao=='1': #análise de erros
            Nummet.metstab()
            print('Digite os índices dos métodos a serem analisados na forma: m0 m1 m2. Por exemplo: "0 1 8 17"\n')
            mets = Nummet.inlist()
            strexata = input('Digite a solução exata da equação diferencial. Por Exemplo: "e**sin(x)"\n')
            a, b, c = Nummet.multerror(data, strexata, mets)
            Nummet.errtable(a, b, c)
            a, b, c = Nummet.multmetnoplot(data, mets)
            Nummet.ptstableplus(a, b, c, strexata)
        elif opcao=='2':#plot e análise
            Nummet.metstab()
            print('Digite os índices dos métodos a serem estudados na forma: "m0 m1 m2". Por exemplo: "0 1 8 17"\n')
            mets = Nummet.inlist()
            strexata = input('Digite a solução exata da equação diferencial. Por Exemplo: "e**sin(x)"\n')
            a, b, c = Nummet.multerror(data, strexata, mets)
            Nummet.errtable(a, b, c)
            a, b, c = Nummet.multmetnoplot(data, mets)
            Nummet.ptstableplus(a, b, c, strexata)
            Nummet.multplotex(data, strexata, mets)
    if cont==0: break
print("Até mais...")



