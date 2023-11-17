# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:32:54 2023

@author: Sebastiano Tomasi
"""
import numpy as np
from numpy import sqrt

import scipy as sp
import scipy.interpolate

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import sys
sys.path.append("../data_modules")
from data_classes import friedmann_sol
import cosmological_functions as cosm_func
import simulation_parameters as params

sys.path.append("../utility_modules")
import numerical_methods as mynm
import plotting_functions as mypl

#%% Global variables

asterisks_lenght=int(60)

"""Create the class object to store the results"""
result=friedmann_sol()

#%%

def solve(time_domain=False,does_print_info=False):
    """This function solve for the background equations, given the parameters in the simulation_parameters module.
        input:
            - time_domain if True, computes all the bakground quantities also as function of time. 
            - doese_print_info if True prints the age of the universe and the deceleration parameter."""
            
    """Compute the dark density evolution"""
    dark_density_evolution_numerical_a=cosm_func.solve_dark_density_evolution_numerical_a()
    scale_parameter_values=dark_density_evolution_numerical_a[0]
    dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)
    
    
    rescaled_hubble_function_a=cosm_func.create_rescaled_hubble_function_a(dark_density_evolution_a)
    
    """Compute also the de_eos numerically in order to plot it."""
    de_eos_numerical_a=np.array([scale_parameter_values,params.de_eos_a(scale_parameter_values)])
    
    
    """Effective eos"""
    result.effective_eos_numerical_a=[scale_parameter_values,
                                      (params.de_eos_a(scale_parameter_values)*dark_density_evolution_numerical_a[1]+  \
                                       cosm_func.rad_density_evolution_a(scale_parameter_values)/3)/    \
                                          (cosm_func.matter_density_evolution_a(scale_parameter_values)+   \
                                         dark_density_evolution_numerical_a[1]+   \
                                        cosm_func.rad_density_evolution_a(scale_parameter_values))]
    
    """Approximate dark density evolution using a step transition."""
    aux=[]
    for i in scale_parameter_values:
        aux.append(cosm_func.step_dark_density_evolution_a(i))
    appox_dark_density_evolution_numerical_a=np.array([scale_parameter_values,aux])
    
    """Rescaled hubble function H/H0, we need it to compute the density parameters"""
    rescaled_hubble_functions_numerical_a=np.array([scale_parameter_values,
                                          rescaled_hubble_function_a(scale_parameter_values)])
    
    dark_density_parameter_numerical_a=np.array([scale_parameter_values,
              dark_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    matter_density_parameter_numerical_a=np.array([scale_parameter_values,
              cosm_func.matter_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    rad_density_numerical_a=np.array([scale_parameter_values,
              cosm_func.rad_density_evolution_a(scale_parameter_values)/rescaled_hubble_functions_numerical_a[1]**2])
    
    """Saving result in the Friedmann results class"""
    result.dark_density_evolution_numerical_a=dark_density_evolution_numerical_a
    result.appox_dark_density_evolution_numerical_a=appox_dark_density_evolution_numerical_a
    result.rescaled_hubble_functions_numerical_a=rescaled_hubble_functions_numerical_a
    result.dark_density_parameter_numerical_a=dark_density_parameter_numerical_a
    result.matter_density_parameter_numerical_a=matter_density_parameter_numerical_a
    result.rad_density_numerical_a=rad_density_numerical_a
    result.de_eos_numerical_a=de_eos_numerical_a
    
    if time_domain:
        """Integrate the friedmann equation"""
        friedmann_equation_integrand=cosm_func.create_log_friedmann_equation_integrand(dark_density_evolution_a)
        hubble_constant_times_t = mynm.integrate(f=friedmann_equation_integrand,
                                                  a=np.log(params.a_min), b=np.log(params.a_max),
                                                  atol=params.friedmann_atol,rtol=params.friedmann_rtol,max_step=params.friedmann_max_stepsize)
        hubble_constant_times_t[0]=np.exp(hubble_constant_times_t[0])#It is y=ln(a)
        scale_parameter_values=hubble_constant_times_t[0]
    
        """Compute t(a)==time_a"""
        time=hubble_constant_times_t[1]/params.hubble_constant
        universe_age = time[-1]#In giga yeras
        
        """Invert t(a) to obtain a(t). In more detail we have a(t) s.t a(t0)=1 
        where t_0 is the age of the universe"""
        scale_parameter_numerical_t=np.array([time,scale_parameter_values])
        
        """Compute the hubble function H(t)=\dot{a}/a and the second derivative of a"""
        scale_parameter_t=sp.interpolate.interp1d(time, scale_parameter_values,
                                          fill_value="extrapolate", assume_sorted=True,kind="quadratic")
        h=mynm.rms_neigbour_distance(time)#Compute average spacing between time points
        new_lenght=int(2/h)#Use h to obtain the new time array lenght
        time=np.linspace(time[0],time[-1],new_lenght)#Build a new time array with equally spaced points
        scale_parameter_values=scale_parameter_t(time)#Compute the corresponding scale parameter values
        
        

        
        scale_parameter_derivative_numerical_t=mynm.derivate(scale_parameter_t,time[0],time[-1], new_lenght)
        
        hubble_function_numerical_t=np.array([time,scale_parameter_derivative_numerical_t[1]/scale_parameter_values])
    
        """We can now calculate the deceleration parameter today: q_0"""
        scale_parameter_derivative_t=sp.interpolate.interp1d(scale_parameter_derivative_numerical_t[0], 
                                                            scale_parameter_derivative_numerical_t[1],
                                            fill_value="extrapolate", assume_sorted=True,kind="quadratic")
        scale_parameter_2derivative_numerical_t=mynm.derivate(scale_parameter_derivative_t,time[0],time[-1], new_lenght)
        
        dec_param_now=-scale_parameter_2derivative_numerical_t[1][-1]/params.hubble_constant**2

        """Save results"""
        result.scale_parameter_numerical_t=scale_parameter_numerical_t
        result.hubble_function_numerical_t=hubble_function_numerical_t
        result.universe_age=universe_age
        result.deceleration_parameter=dec_param_now
        
        """Compare the age of the universe of the model to LCDM"""
        if does_print_info:
            if params.omega_matter_now==1:
                theoretical_universe_age=2/(3*params.hubble_constant)
            else:
                theoretical_universe_age= cosm_func.time_a_LCDM(1)
            print("*"*asterisks_lenght)
            print("MODEL: t0=",round(universe_age,5)," Gy\t\tLCDM: t0=",round(theoretical_universe_age,5)," Gy")
            print("*"*asterisks_lenght)
            print("MODEL: q0 = ",round(dec_param_now,3),"""\t\t\tLCDM: q0 = """,round(params.omega_matter_now/2-params.omega_dark_now,3))
            print("*"*asterisks_lenght)
    return result









