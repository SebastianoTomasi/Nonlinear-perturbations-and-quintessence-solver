# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:47:23 2022

@author: Sebastiano Tomasi
"""
import numpy as np
import scipy as sp
import scipy.integrate

import sys
sys.path.append("../utility_modules")
import plotting_functions as mypl
import lcdm_model
import cosmological_functions as cosm_func
import import_export as myie

# def f(x):
#     return np.sin(x)
# sol=integrate_(f,0,10,2,-1)
# mypl.plot(sol,func_to_compare=lambda x:-np.cos(x),dotted=True)
    
def integrate_symps(a,b,N,f):
    """Compute the intrgral function in the interval [a,b] with 
    number of steps N of the function f with the sympson's method.
    Returns [x,F(x)] where F is the integral func of f. """
    if N%2!=0:
        raise Exception("N must be even!")
    if a<b:
        flip=False
    else:
        flip=True
    
    h=(b-a)/N
    integrale=0
    x=a
    res=[[],[]]
    for i in range(int(N+1)):
        res[0].append(x)
        res[1].append(integrale)
        integrale = integrale + h*(f(a + i*h) +4*f(a+(2*i+1)*h/2) + f(a + (i+1)*h) )/6
        # integrale= integrale +h*(f(a+i*h)+f(a+(i+1)*h))/2#trapezi
        x=x+h
    if flip:
        res[0]=np.flip(res[0])
        res[1]=np.flip(res[1])
        return np.array(res)
    
    return np.array(res)



def Nderivate(f):
    """Numerical derivative:
        input:
            - f=[[x_1,...,x_n],[y_1,...,y_n]]
        output:
            - [[x_1,...,x_n],[y'_1,...,y'_n]]"""

    x=f[0]
    y=f[1]
    derivative=[]
    for i in range(int(len(x)-1)):
        derivative.append( (y[i+1]-y[i])/(x[i+1]-x[i]))
    derivative.append(derivative[-1])#fill the missing value with an Euler step
    return np.array([x,derivative])

def derivative_x0(f,h,x0):
    """Derivative of a callable function:
        input:
            - f callable
            - h step
            - x0 point at which the derivative is computed
        output:
            - derivative_at_x0 = f'(x0)"""
    derivative_at_x0=(f(x0+h/2)-f(x0-h/2))/h
    return derivative_at_x0

def derivate(f,a,b,n):
    """Derivative of a callable function:
        input:
            - f callable
            - a left domain extreme
            - b right domain extreme
        output:
            - [[x_1,...,x_n],[f'_1,...,f'_n]]"""
    h=(b-a)/n
    derivative=[[],[]]
    for i in range(int(n)):
        x0=a+i*h
        derivative[0].append(x0)
        derivative[1].append(derivative_x0(f,h,x0))
        
    return derivative        
    

# def integrate(f,a,b,n,Fa=0):
#     """Find the integral function of f in the range [a,b]:
#         input:
#             - f callable
#             - [a,b] integration interval
#             - n nuber of points used
#             - Fa is the value of the integral function at x=a (F(a))."""
#     def fun(t,y):
#         return np.array([f(t)])
#     """Perform the integration with rk4"""
#     rk4_result=rk4(fun,t_span=(a,b),y0=[Fa],n=n)
    
#     if a<b:#Flip the results if the order is wrong.
#         flip=False
#     else:
#         flip=True
#     if flip:
#         rk4_result[0]=np.flip(rk4_result[0])
#         rk4_result[1]=np.flip(rk4_result[1])
#         return np.array(rk4_result)
#     return rk4_result

def integrate(f,a,b,Fa=0,rtol=1e-3,atol=1e-6,max_step=np.inf):
    """Find the integral function of f in the range [a,b]:
    #         input:
    #             - f callable
    #             - [a,b] integration interval
    #             - n nuber of points used
    #             - Fa is the value of the integral function at x=a (F(a))."""
    def fun(t,y):
        return np.array(f(t))
    sol=sp.integrate.solve_ivp(fun,t_span=(a,b), y0=[Fa],
                                      method="RK45",rtol=rtol,atol=atol,max_step=max_step)
    x=sol.t
    y=sol.y[0]
    flip=a>b
    if flip:
        x=np.flip(x)
        y=np.flip(y)
        return [x,y]
    return [x,y]


def rk4(f,t_span,y0,n):
    """Description of the algorithm: 

        dy / dt = f(t, y)
        y(t0) = y0

    Here t is a 1-D independent variable (time), y(t) is an
    N-dimensional vector-valued function (state), and an N-dimensional
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0 in the interval t_span=(t_min,t_max) 
    using n as number of points"""
    n=int(n)
    y0=np.array(y0)
    t_min=t_span[0]
    t_max=t_span[1]
    dt=(t_max-t_min)/n
    
    t=t_min#Cycle variables
    y=np.array(y0)
    
    time_axis=np.array([t])#Result conteiners
    rk4=np.array([[x] for x in list(y0)])
    for i in range(n):
        k1=f(t,y)
        k2=f(t + dt/2 ,y + dt*k1/2)
        k3=f(t + dt/2 ,y + dt*k2/2)
        k4=f(t + dt ,y + dt*k3)
        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        t = t + dt 
        time_axis = np.append(time_axis,t)
        rk4 = np.concatenate((rk4,np.array([[x] for x in y])),axis=1)
    
    result = np.concatenate((np.array([time_axis]),rk4))
    return result

def Nequation_solve(f,c,tol=1e-9):
    """Solves a numerical equation:
        input:
            - f is a numpy array array 
            - c a constant.
        output:
            -index is the index such f[index]=c"""
    x=np.array(f[0])
    y=np.array(f[1])
    difference=y-c
    lenght=len(difference)
    for i in range(lenght-1):
        diff0=difference[i]
        diff1=difference[i+1]
        if abs(diff0)<=tol:
            """In this case the root is between exactly at i+1"""
            return x[i]
        if abs(diff1)<=tol:
            """In this case the root is between exactly at i+1"""
            return x[i+1]
        if diff0>0 and diff1<0 or diff0<0 and diff1>0:
            """In this case the root is between x[i] and x[i+1]"""
            return x[i]
    print("WARNING!!!!!!!!!!!!")
    print("The solution could be outside the array.\n The difference between f[-1] and the point is:"+"%.1e" % diff1)
    return -1



def bisection(f,a,b,tol):
    """Use the bisection algorithm to find the root of f between a and b and stop when
    |f(m)|<tol"""
    m,f_m=0,0
    f_a,f_b=f(a),f(b)
    
    if np.sign(f_a)==np.sign(f_b):
        raise Exception("The scalars a and b do not bound a root")

    while True:
        m=(a+b)/2
        # print(m)
        f_m=f(m)
        if abs(b-a)<=tol:
            return m
        if np.sign(f_a)==np.sign(f_m):
            a=m
            f_a=f_m
        if np.sign(f_b)==np.sign(f_m):
            b=m
            f_b=f_m


def bisection_(infty_minus_nonlinear_delta_c,a,b,tol,a_coll):
    """Bisection alghoritm used to find the root of infty_minus_nonlinear_delta_c between a and b. 
    Stop when |b-a|<tol. Designed to compute specifically delta_collapse_star"""
    m,f_m=0,0
    f_a,f_b=infty_minus_nonlinear_delta_c(a,a_coll)[0],infty_minus_nonlinear_delta_c(b,a_coll)[0]
    
    if np.sign(f_a)==np.sign(f_b):
        raise Exception("The scalars a and b do not bound a root")

    while True:
        m=(a+b)/2
        # print(m)
        f_m, density_contrast = infty_minus_nonlinear_delta_c(m,a_coll)
        if abs(b-a)<tol:
            return [m,density_contrast]
        if np.sign(f_a)==np.sign(f_m):
            a=m
            f_a=f_m
        if np.sign(f_b)==np.sign(f_m):
            b=m
            f_b=f_m


def logspace(start,stop,num):
    """Generates a sequence of logaritmically spaced values 
    ùof lenght num from start to stop """
    sequence=[]
    alfa=1/(num-1)*np.log10(stop/start)#logarithmic spacing
    for n in range(int(num)):
        sequence.append(10**(n*alfa)*start)
    return np.array(sequence)




def rms_normalized_difference(f_num,f_anal):
    """Returns the root mean square difference between f_num which is a numerical function [[x_i],[y_i]]
    and a callable function f_anal"""
    N=len(f_num[0])
    error=0
    function_average=0
    for i in range(N):
        x_i=f_num[0][i]
        y_i=f_num[1][i]
        f_x_i=f_anal(x_i)
        error+= (f_x_i-y_i)**2
        function_average+=y_i**2
    error=np.sqrt(error/N)
    function_average=np.sqrt(function_average/N)
    print("error/func_average=",error/function_average)
    return np.sqrt(error/N)


def rms_neigbour_distance(x):
    """Returns the variance of x[i+1]-x[i], where x is an array of values."""
    N=len(x)
    variance=0
    for i in range(N-1):
        variance+= (x[i+1]-x[i])**2
    return np.sqrt(variance/(N-1))
        



def find_n_h(a,b,h):
    """Find the closest integer n for which H=|b-a|/n is closest to the desired value h"""
    interval_lenght=abs(b-a)
    if h>interval_lenght:
        raise Exception(f"The stepsize h={h} is greater then then the interval lenght |b-a|={abs(b-a)}")
    done = False
    n,H,H_prec=0,interval_lenght,interval_lenght
    while not done:
        H_prec=H
        n+=1
        H=interval_lenght/n
        if H<=h and H_prec>=h:
            done=True
        
    return n,H



def mooving_average(data,window_size):
    if window_size>len(data):
        raise Exception("The window size {window_size} must be less then {len(data)}")
    # Create a window with equal weights
    window = np.ones(window_size)/window_size
    
    # Compute moving average using convolution
    filtered_data = np.convolve(data, window, 'valid')
    return filtered_data

    


