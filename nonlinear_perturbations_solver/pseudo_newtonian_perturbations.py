# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:39:15 2023

@author: Sebastiano Tomasi
"""

import numpy as np
import scipy as sp

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func

sys.path.append("../utility_modules")
import numerical_methods as mynm

#%% Global variables
#Where the integration starts for both solve_nonlinear_perturbations and solve_growth_factor





#%%
"""Definition of some useful functions."""

def virialization_radius(zeta_ta,a_ta,a_c):
    """Implementation of the formula to find the virialization radius"""
    eta_t=2/zeta_ta*omega_dark_a(a_ta)/omega_matter_a(a_ta)
    eta_v=2/zeta_ta*(a_ta/a_c)**3*omega_dark_a(a_c)/omega_matter_a(a_c)
    x_vir=(1-eta_v/2)/(2+eta_t-3*eta_v/2)
    return x_vir
    
def find_virialization_scale(nonlinear_density_contrast_a,zeta_ta,a_ta,a_c):
    """
    Input:
        -nonlinear_density_contrast_a: callable
        -zeta_ta: overdensity at turn around
        -a_ta: scale parameter at turn around
        -a_c: scale parameter at collapse
    Output:
        -[a_vir,x_vir]: scale parameter at virialization and sphere radius at virialization.
    """
    x_vir=virialization_radius(zeta_ta,a_ta,a_c)
    def fun_to_find_zeros(a):
        fun_to_find_zeros=x_vir-zeta_ta**(1/3)*a/(a_ta*(1+nonlinear_density_contrast_a(a))**(1/3))
        return fun_to_find_zeros
    
    a_vir=mynm.bisection(fun_to_find_zeros, a_ta, 1, tol=params.virialization_scale_tol)
    return [a_vir,x_vir]

def infty_minus_nonlinear_delta_c(delta_c_star,a_coll):
    """Difference between the maxium nonlinear matter density contrast reached and the numerical infinity,
    the goal is to minimize this difference at a_collapse.
    Input:
        -delta_c_star: the initial condition.
        -a_coll: scale parameter at which we want the collapse to happens.
    Output:
        [difference,nonlinear_density_contrast]"""
    nonlinear_density_contrast=solve_nonlinear_perturbations(delta_c_star,a_coll)
    difference=params.numerical_infty-nonlinear_density_contrast[1][-1]
    return [difference,nonlinear_density_contrast]

def find_delta_collapse_star(collapse_scales_a,a=1.68,b=3.3,exact_decimals=params.exact_decimals):
    """
    Finds the initial conditions delta_c^* that result in the sphere collapsing at a_c
    Input: 
        -collapse_scales_a: can be a number or an array of scale parameter values in the range [0,1].
        -a,b: define the interval [a,b] where the solution is found through the bisection alghorithm
        -exact_decimals: exact decimals of the initial condition delta_c^* 
    Output:
        [delta_coll_stars,density_contrast] where density_contrast is a list that contains all the solutions
            returned by infty_minus_nonlinear_delta_c"""
            
    """Check if only 1 value is given for collapse_scales_a"""
    return_scalar = False
    if isinstance(collapse_scales_a, (int, float)):#If collapse_scales_a is an int or float
        collapse_scales_a = [collapse_scales_a]
        return_scalar = True

    delta_coll_stars=[]
    density_contrasts=[]
    for a_coll in collapse_scales_a:
        delta_coll_star,density_contrast=mynm.bisection_(infty_minus_nonlinear_delta_c,a,b,10**(-exact_decimals-1),a_coll)
        
        """We save the nonlinear matter density contrast solutions to return them."""
        density_contrasts.append(density_contrast)
        
        delta_coll_star=round(delta_coll_star,exact_decimals+1)
        delta_coll_stars.append(delta_coll_star)
        print("a_c=",round(a_coll,4),"     delta_c_star=",round(delta_coll_star,exact_decimals))
        
    if return_scalar:
        return [delta_coll_stars[0],density_contrasts]
    else:
        return [delta_coll_stars,density_contrasts]
    
    
def find_turn_around_scale(nonlinear_density_contrast_a,a_coll):
    """Find the turn around scale by minimizing -x, where x is the radius of the sphere.
    Input:
       -nonlinear_density_contrast_a: callable
       -a_coll: scale parameter at collapse
       -tol: tolerance used in the minimization alghoritm
    Output:
       a_ta: scale parameter at turn around
       """
    def fun_to_minimize(a):
        """This function is proportional minus the sphere radius"""
        fun_to_minimize=-a/(1+nonlinear_density_contrast_a(a))**(1/3)
        return fun_to_minimize
    
    bracket_interval=[1e-5,(0.5)**(2/3)*a_coll,a_coll]
    optimize_result=sp.optimize.minimize_scalar(fun_to_minimize,bracket=bracket_interval,  tol=params.turn_around_scale_tol)
    a_ta=round(optimize_result.x,int(-np.log10(params.turn_around_scale_tol))+1)
    return a_ta


#%%
"""Definition of the ODE system to solve: """

def nonlinear_density_contrast_eq(a,y):
    """This function defines the NONLINEAR matter density contrast differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\theta 
    y[2]=\delta_de
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\theta'
    fun[2]=delta_de'
    """
    if params.does_de_cluster:
        fun=[0]*3
        fun[0] = -(1+y[0])*y[1]/a
        fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-y[1]**2/(3*a)-3/(2*a)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*(1+3*params.c_eff(a))*y[2])
        fun[2] = -3/a*(params.c_eff(a)-params.de_eos_a(a))*y[2]-(1+params.de_eos_a(a)+(1+params.c_eff(a))*y[2])*y[1]/a
        if y[2]<-1:#Positive dark energy density implies y[2]=delta_de>-1
            y[2]=-1
    elif not params.does_de_cluster:
        fun=[0]*2
        fun[0] = -(1+y[0])*y[1]/a
        fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-y[1]**2/(3*a)-3/(2*a)*(omega_matter_a(a)*y[0])
    

    return np.array(fun)



def linear_density_contrast_eq(a,y):
    """This function defines the LINEAR matter density contrast differential equation whene dark energy perturbations are set to zero,
    but the effects of dark energy are still considered in the background.
    y[0]=\delta_m
    y[1]=\theta 
    y[2]=\delta_de
    
    fun[0]=y'[0]=\delta_m'
    fun[1]=y'[1]=\theta'
    fun[3]=delta_de'
    """
    if params.does_de_cluster:
        fun=[0]*3
        fun[0] = -y[1]/a
        fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-3/(2*a)*(omega_matter_a(a)*y[0]+omega_dark_a(a)*(1+3*params.c_eff(a))*y[2])
        fun[2] = -3/a*(params.c_eff(a)-params.de_eos_a(a))*y[2]-(1+params.de_eos_a(a))*y[1]/a
        if y[2]<-1:#Positive dark energy density implies y[2]=delta_de>-1
            y[2]=-1
    elif not params.does_de_cluster:
        fun=[0]*2
        fun[0] = -y[1]/a
        fun[1] = -(1-3*effective_eos_a(a))/(2*a)*y[1]-3/(2*a)*(omega_matter_a(a)*y[0])
        
    return np.array(fun)



#%%
def solve_nonlinear_perturbations(delta_coll_star,a_coll):
    """Solves the nonlinear perturbation equations of the pseudo newtonian approximation
    both for clustering or not clutering dark energy.
    Input:
        -delta_coll_star: is related to the initial condition for delta_m
        -a_coll: the scale parameter at which the collapse happens, that is when delta_m=params.numerical_infty
    Output:
        -nonlinear_matter_density_contrast_numerical_a: a quite explicative name."""
    a_min= params.a_ini#Left extreme of the integration domain.
    a_max= a_coll#Right extreme of the integration domain.
    n=round(0.25*(-1+np.sqrt(24*omega_matter_a(a_min)+(1-3*effective_eos_a(a_min))**2)+3*effective_eos_a(a_min)),4)#Growth factor exponent at early times.
    delta_m_ini=delta_coll_star*(a_min/a_coll)**n#delta_m initial condition.
    theta_min=-n*delta_m_ini#Initial condition for the theta function.
    if params.does_de_cluster:
        delta_de_ini=n*(1+params.de_eos_a(a_min))*delta_m_ini/(n+3*(params.c_eff(a_min)-params.de_eos_a(a_min)))#delta_de initial condition.
        init_cond=[delta_m_ini,theta_min,delta_de_ini]# Initial conditions for clustering dark energy, to pass to scipy solver.
    elif not params.does_de_cluster:
        init_cond=[delta_m_ini,theta_min]# Initial conditions for not clustering dark energy.
    
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(nonlinear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=params.nonlinear_perturbations_atol,rtol=params.nonlinear_perturbations_rtol)
    
    """Extract the results from the rk4 data structure"""
    nonlinear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    
    if params.does_de_cluster:
        nonlinear_de_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[2]])
    elif not params.does_de_cluster:
        pass
    
    return nonlinear_matter_density_contrast_numerical_a #[nonlinear_matter_density_contrast_numerical_a,nonlinear_de_density_contrast_numerical_a]



def solve_growth_factor(delta_coll_star,a_coll,a_min=params.a_ini):
    """Solves the linear perturbation equations of the pseudo newtonian approximation
    both for clustering or not clutering dark energy.
    Input:
        -delta_coll_star: is related to the initial condition for delta_m
        -a_coll: the scale parameter at which the collapse happens, that is when delta_m=params.numerical_infty
    Output:
        -linear_matter_density_contrast_numerical_a: a quite explicative name."""
    #a_min Left extreme of the integration domain.
    a_max= a_coll#Right extreme of the integration domain.
    n=round(0.25*(-1+np.sqrt(24*omega_matter_a(a_min)+(1-3*effective_eos_a(a_min))**2)+3*effective_eos_a(a_min)),4)#Growth factor exponent at early times.
    delta_m_ini=delta_coll_star*(a_min/a_coll)**n#delta_m initial condition.
    theta_min=-n*delta_m_ini#Initial condition for the theta function.
    if params.does_de_cluster:
        delta_de_ini=n*(1+params.de_eos_a(a_min))*delta_m_ini/(n+3*(params.c_eff(a_min)-params.de_eos_a(a_min)))#delta_de initial condition.
        init_cond=[delta_m_ini,theta_min,delta_de_ini]# Initial conditions for clustering dark energy, to pass to scipy solver.
    elif not params.does_de_cluster:
        init_cond=[delta_m_ini,theta_min]# Initial conditions for not clustering dark energy, to pass to scipy solver.
    
    
    """Perform the integration """
    rk4_result=sp.integrate.solve_ivp(linear_density_contrast_eq,t_span=(a_min,a_max), y0=init_cond, 
                                      method="RK45",atol=params.growth_factor_atol,rtol=params.growth_factor_rtol)
    
    """Extract the needed informations from the rk4 result"""
    linear_matter_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[0]])
    if params.does_de_cluster:
        linear_de_density_contrast_numerical_a=np.array([list(rk4_result.t),rk4_result.y[2]])
    else:
        linear_de_density_contrast_numerical_a=np.array([np.array(list(rk4_result.t)),np.zeros(len(linear_matter_density_contrast_numerical_a[0]))])
    
    return [linear_matter_density_contrast_numerical_a,linear_de_density_contrast_numerical_a]



#%%

def solve(friedmann_solution=None,only_linear=False):
    
    """Use the friedmann solution to get the effective eos,  omega matter and omega_dark"""
    aux=friedmann_solution.effective_eos_numerical_a
    global effective_eos_a
    effective_eos_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)

    aux=friedmann_solution.matter_density_parameter_numerical_a
    global omega_matter_a
    omega_matter_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)

    aux=friedmann_solution.dark_density_parameter_numerical_a
    global omega_dark_a
    omega_dark_a=sp.interpolate.interp1d(aux[0], aux[1],
                                        fill_value="extrapolate", assume_sorted=False)
    
    linear_density_contrast=solve_growth_factor(1,1,params.scale_at_lss)
    if only_linear:
        return [None,None,linear_density_contrast]
    else:
        """Define when we want the collapse to happen: it is decided with [params.min_redshift,params.max_redshift] and the input params.number_of_points"""
        collapse_scales_z=np.linspace(params.min_redshift,params.max_redshift,params.number_of_points)
        collapse_scales_a=cosm_func.a_z(collapse_scales_z)
        
        """Compute the delta collapse star, that are the initial conditions 
        to obtain the collapse at the collapse scales. Find_delta_collapse_star also returns 
        the nonlinear matter density contrast."""
        delta_c_stars,nonlinear_density_contrasts=find_delta_collapse_star(collapse_scales_a)
        
        
        """Compute the linear density contrast at collapse and the growth factor for each collapse time"""
        linear_density_contrast_at_collapse=[[],[]]
        for i in range(params.number_of_points):
            """Compute the grow factor,i.e., the linear density contrast without decaying mode."""
            growth_factor=solve_growth_factor(delta_c_stars[i],collapse_scales_a[i])[0]
           
            """Fill the list that stores the linearized matter density contrast at collapse"""
            linear_density_contrast_at_collapse[0].append(collapse_scales_z[i])
            linear_density_contrast_at_collapse[1].append(growth_factor[1][-1])
        
        
        """Compute the overdensity at virialization defined in the theis as zeta_m^*"""
        virialization_overdensities_stars=[[],[]]
        for i in range(params.number_of_points):
            """Interpolate the i-th nonlinear density contrast that correspond to the i-th collapse time"""
            nonlinear_density_contrast_a=sp.interpolate.interp1d(nonlinear_density_contrasts[i][0], nonlinear_density_contrasts[i][1],
                                                fill_value="extrapolate", assume_sorted=False)
            
            """Find the turn around scale parameter and compute the turn around matter overdensity"""
            turn_around_scale=find_turn_around_scale(nonlinear_density_contrast_a,collapse_scales_a[i])
            turn_around_overdensity=np.round(nonlinear_density_contrast_a(turn_around_scale),4)+1
            
            """Find the virialization scale parameter and the radius at virialization x_vir to compute zeta_m^*"""
            virialization_scale,x_vir=find_virialization_scale(nonlinear_density_contrast_a,turn_around_overdensity,
                                                               turn_around_scale, collapse_scales_a[i])
            virialization_overdensities_stars[0].append(collapse_scales_z[i])
            virialization_overdensities_stars[1].append(turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)

        """Debug prints"""
        # print("32=",round((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,3))
        # print("a_c=",round(collapse_scales_a[i],3),"    a_ta=",round(turn_around_scale/collapse_scales_a[i],5),"    x_vir=",round(x_vir,3))
        # print((collapse_scales_a[i]/(turn_around_scale*x_vir))**3,"*",\
        #       turn_around_overdensity,"=",turn_around_overdensity*(collapse_scales_a[i]/(turn_around_scale*x_vir))**3)
        return [linear_density_contrast_at_collapse,virialization_overdensities_stars,linear_density_contrast]



