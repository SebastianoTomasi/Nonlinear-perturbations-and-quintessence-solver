# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:18:25 2023

@author: sebas
"""
import numpy as np
from numpy import pi



import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)


import dark_energy_eos as eos
import constants as const

#%%

pseudo_newtonian_perturbations_run_params = {
    "does_de_cluster": True,
    "save_data": True,
    # If time_domain=True, it computes t_0 and q_0 and prints them.
    "time_domain": True,
    "number_of_points":50,  # Number of data points computed for delta_c and zeta_vir_star
    "varied_parameter": "trans_steepness",  # can be = 'w_i';'w_f';'trans_steepness';'trans_z';"w" for wCDM
    #If you set var_par_values to None, the varied parameter is varied in np.linspace(init_value,final_value,n_samples).
    #If var_par_values is a list, the varied parameter assume the var_par_values.
    # "varied_par_values":[0.1,0.8,2],
    "varied_par_values":None,
    # "varied_par_values":[0.5,1,5],
    # "varied_par_values":[-0.5,-1.5,-2.5],
    "init_value": -0.5,
    "final_value": -0.9,
    "n_samples": 1
}

dark_energy_eos_params={
    #Equation of state for fast transition dark energy models.
    #Select the desired dark energy equation of state. You can find all the equations and 
    #define new ones in the dark_energy_eos module.
    #selected_de_eos can be wcdm,cpl,de_eos_1,...,de_eos_6
    "selected_de_eos": "de_eos_6",
    "w_i": -0.4,
    "w_f": -1,
    "trans_steepness": 10,# gamma_values=[[10,[0.5,1,5]],
                              # [5,[0.5,1.5,3]],
                              # [0.1,[0.3,0.8,2]],
                              # [0.04,[0.1,0.2,0.3]],
                              # [-1,[-0.5,-1.5,-2.5]]]
    "trans_z": 0.5,
    #If selected_de_eos=wcdm, it uses only w while if selected_de_eos=cpl uses w_i and w_f 
    "w": -1,
    #Effective sound speed, can be defined as a function. You can use lambda functions directly 
    #in this dictionary, or you can define a function otuside and set c_eff_value equal to it.
    "c_eff_value": 0,
    }

halo_mass_function_run_params={
    "fitting_func":"ST",# PS, ST
    "window_function":"TH",#TH GAUS
    "z":[0,0.5,1],#Redshifts at which the mass function is computed.
    "this_run_specifier_1": "perturbed_de",#[ "unperturbed_de", "perturbed_de", "LCDM", "EDS"]
    "this_run_specifier_2": "de_eos_1",#wCDM,LCDM,de_eos_1,...,de_eos_6
    "this_run_specifier_3": "w_i",#['w_i','w_f','trans_steepness','trans_z']
    #Rrange for the mass function
    "min_mass":1e10,
    "max_mass":1e14,
    "number_of_masses":500,
    "min_k":2e-5,
    "max_k":5
    }

cosmological_params={
    "omega_matter_now": 0.3,
    "omega_rad_now": 0,
    #"omega_rad_now": round(0.2473/(67.4)**25),
    "hubble_constant_standard_units": 67,
    "scale_at_lss": 1/(1+1090.0),
    #Range of integration: domain of the cosmological time t(a)."
    "a_min": 1e-10,
    "a_max": 1,
    #Define the domain for the density contrast and overdensity.
    "min_redshift": 0,
    "max_redshift": 20,
    "spectral_index": 0.96,
    "sigma8":0.85
    }



precision_params={
    #Tolerance in the bisection method used to compute the virialization scale.
    "virialization_scale_tol":1e-6,
    #Tolerance for scipy.optimize.minimize_scalar() to compute the turn around scale.
    "turn_around_scale_tol":1e-6,
    #Initial time for both solve_nonlinear_perturbations and solve_growth_factor.
    "a_ini":cosmological_params["a_min"],
    #Precison parameters for scipy.integrate.solve_ivp() for nonlinear and linear perturbations.
    "nonlinear_perturbations_atol":1e-10,
    "nonlinear_perturbations_rtol":1e-8,    
    "growth_factor_atol":1e-10,
    "growth_factor_rtol":1e-8,
    #Precision parameters for the integral of the fluid equation.
    "dark_density_evolution_atol":1e-10,
    "dark_density_evolution_rtol":1e-8,
    "dark_density_evolution_max_step":1e-3,
    #Precision parameters for the integral of the friedmann equation to compute t(a).
    "friedmann_atol":1e-10,
    "friedmann_rtol":1e-8,
    "friedmann_max_stepsize":1e-3,
    #Exact decimals of linearly extrapolated matter density contrast at collapse.
    "exact_decimals": 5,
    #Numerical infinity, used to stop the integration of the nonlinear density contrast differential 
    #equations. It stops when the nonlinear delta_matter>numerical_infty.
    "numerical_infty": 1e11,
    #Precision parameters for computing the halo mass function.
    #Those two parameters are for the integral to compute the normalization of the power spectrum.
    "normalization_integrand_atol":1e-8,
    "normalization_integrand_rtol":1e-6,
    #Precision parameters for the integral to compute the variance of the filitered density contrast
    "mass_variance_filtered_spectrum_atol":1e-6,
    "mass_variance_filtered_spectrum_rtol":1e-4,
    #Precision parameters for the integral for the logarithmic derivative of log(sigma).
    "log_der_of_log_sigma_atol":1e-6,
    "log_der_of_log_sigma_rtol":1e-4
    }

def declare_dict(D):
    """Declare the variables stored in D as key=value."""
    for key,value in D.items():
        globals()[key]=value
        
declare_dict(pseudo_newtonian_perturbations_run_params)
declare_dict(halo_mass_function_run_params)
declare_dict(cosmological_params)
declare_dict(dark_energy_eos_params)
declare_dict(precision_params)

#%% DERIVED PARAMETERS

scale_at_equivalence=1/(2.4*omega_matter_now*hubble_constant_standard_units**2)

omega_dark_now = 1-omega_matter_now-omega_rad_now

hubble_constant = hubble_constant_standard_units*1e3*const.Gy/const.Mpc# Gy-1
"""H[s]=H[Gy]/Gy -> H in sec. = H in Gy divided by 1Gy in seconds"""

small_h=hubble_constant_standard_units/100#Dimensionless

critical_density = 3*hubble_constant**2/(8*pi*const.G*const.Gy**2)#kg m-3
critical_density_Ms_Mpc=2.775*1e11#Ms/Mpc^3*h^2

rho_m0_Ms_Mpc=critical_density_Ms_Mpc*omega_matter_now

mass_to_radius=(3/(4*pi*critical_density_Ms_Mpc*omega_matter_now))**(1/3)#This times M**(1/3) gives the radius in Mpc

radius8=8# Mpc/h
mass8=4/3*np.pi*radius8**3*rho_m0_Ms_Mpc# M_sun
#%%
"""Gets the function eos.selected_de_eos and saves it in de_eos_z"""
de_eos_z = getattr(eos, selected_de_eos)
def de_eos_a(a):
    z=1/a-1
    de_eos_a=de_eos_z(z,w_i=w_i,w_f=w_f,z_t=trans_z,gamma=trans_steepness,w=w)
    """You must input the dark_energy_eos_params as global variables instead of 
    inputting directly the dictionary,because the other files will modify the value of the global variables but not the values
    stored inside the dictionaries. Hence by doing this you can vary the parameters from other modules, """
    return de_eos_a

"""If c_eff is a number, we turn it into a callable function. If it is a function
we leave it as it is."""
if isinstance(c_eff_value, (int,float)):
    def c_eff(a):
        return c_eff_value


















