# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: sebas
"""


import numpy as np
from numpy import pi,log,sqrt,exp

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import simulation_parameters as params

import sys
sys.path.append("../utility_modules")
import numerical_methods as mynm


#%%
"""Matter density parameter in function of a"""
def matter_density_evolution_a(a):
    matter_density_evolution_a= params.omega_matter_now/a**3
    return matter_density_evolution_a

"""Radiation density parameter in function of a"""
def rad_density_evolution_a(a):
    rad_density_evolution_a= params.omega_rad_now/a**4
    return rad_density_evolution_a

"""Finds the function g(a) for the dark energy."""
def solve_dark_density_evolution_numerical_a():
    """Definition of the integrand in the exponent."""
    def de_exponent(x):
        """Since the integral is done w.r.t dx, x=ln(a), the expo"""
        de_exponent = -3*(1+params.de_eos_a(np.exp(x)))
        return de_exponent
    """Compute dark_density_evolution_a(a)=omega_dark_now*g(a) """
    de_exponent_integral=mynm.integrate(f=de_exponent,a=np.log(params.a_max),b=np.log(params.a_min),
                                        Fa=0,rtol=params.dark_density_evolution_atol,
                                        atol=params.dark_density_evolution_rtol,
                                        max_step=params.dark_density_evolution_max_step)
    scale_parameter_values=np.exp(de_exponent_integral[0])
    dark_density_evolution_numerical_a=np.array([scale_parameter_values,params.omega_dark_now*exp(de_exponent_integral[1])])
    
    return dark_density_evolution_numerical_a

"""Create the integrand for computing the cosmological time, also in d[log(a)]"""
def create_friedmann_equation_integrand(dark_density_evolution_a):#Not used 
    E=create_rescaled_hubble_function_a(dark_density_evolution_a)
    def friedmann_equation_integrand(a):
        return 1/(a*E(a))
    return friedmann_equation_integrand

def create_log_friedmann_equation_integrand(dark_density_evolution_a):#Logarithmic integrand
    E=create_rescaled_hubble_function_a(dark_density_evolution_a)
    def friedmann_equation_integrand(y):#take as input y=ln(a)
        return 1/E(np.exp(y))
    return friedmann_equation_integrand

def create_rescaled_hubble_function_a(dark_density_evolution_a):#Usually denoted by E(a)=(H/H0)
    def rescaled_hubble_function_a(a):
        return sqrt(matter_density_evolution_a(a) + dark_density_evolution_a(a)+rad_density_evolution_a(a))
    return rescaled_hubble_function_a

"""Derivative of the dark energy eos"""
def de_eos_derivative_a(a):
    raise Exception("The derivative of the eos is not yet implemented.")
    de_eos_derivative_a=None
    return de_eos_derivative_a

#%%
"""Approximation for the dark density evolution """
def step_eos(a):#Step eos
    a_t=1/(1+params.trans_z)
    if a < a_t:
        return params.w_i
    else:
        return params.w_f
    
def step_dark_density_evolution_a(a):
    a_t=1/(1+params.trans_z)
    if a_t>1:
        raise Exception("trans_z must be a positive number")
    if a>=a_t:
        return params.omega_dark_now*a**(-3*(1+params.w_f))
    if a<a_t:
        return  params.omega_dark_now*a**(-3*(1+params.w_i))*a_t**(-3*(params.w_f-params.w_i))
    


#%%
"""Statistical properties of the universe"""

"""Power spectrum and transfer function"""
def primordial_power_spectrum(k,A):
    return A*k**params.spectral_index

def transfer_function(k):#from Bardeen et al. 1986
    q=k/(params.omega_matter_now*params.small_h**2)
    transfer_func=np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-1/4)
    return transfer_func

def power_spectrum(k,A):
    return transfer_function(k)**2*primordial_power_spectrum(k,A)

"""Window functions"""
def TH_window_function(k,M):
    R=params.mass_to_radius*M**(1/3)
    return 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

def TH_window_function2_derivative(k,M):#Is the derivative of the window function squared
    R=params.mass_to_radius*M**(1/3)
    return (np.sin(k*R)-k*R*np.cos(k*R))*(np.sin(k*R)*(1-3/(k*R)**2)+3*np.cos(k*R)/(k*R))

def GAUS_window_function(k,M):
    R=params.mass_to_radius*M**(1/3)
    return np.exp(-k**2*R**2/2)

def GAUS_window_function2_derivative(k,M):
    R=params.mass_to_radius*M**(1/3)
    return -R**2*k*np.exp(-k**2*R**2/2)

"""Mass function"""
def create_mass_function(f,sigma,growth_factor,delta_c,log_der_of_log_sigma,rho_m):
    def mass_function(M,z):
        nu=delta_c(z)/(sigma(M)*growth_factor(z))    
        return rho_m/M*f(nu)*abs(log_der_of_log_sigma(M))
    return mass_function
    
def f_ps(nu):
    return np.sqrt(2/np.pi)*nu*np.exp(-nu**2/2)

def f_st(nu,a=0.707,q=0.3,A=0.322):
    nu_prime=np.sqrt(a)*nu
    return A*(1+nu_prime**(-2*q))*f_ps(nu)
    
    

#%%
"""Miscellaneous"""

def time_a_LCDM(a):#Age of the universe in LCDM in Gy
    r=params.omega_dark_now/params.omega_matter_now
    time_a_LCDM=-np.log(-2*a**(3/2)*sqrt(r)*sqrt(a**3*r+1)+ 2*a**3*r+1)/(3*sqrt(params.omega_dark_now))
    return time_a_LCDM/params.hubble_constant
#%%
"""Conversion between redshift <-> scale parameter"""
def z_a(a):
    z_a=1/a-1
    return z_a
def a_z(z):
    a_z=1/(1+z)
    return a_z


#%%
""" Useful values """
def delta_c_eds(a):
    return 1.68647019984
def virial_overdensity(a):
    return 146.841245384
def virial_overdensity_star(a):
    return 177.65287922
def turn_aroun_overdensity(a):
    return 5.551652