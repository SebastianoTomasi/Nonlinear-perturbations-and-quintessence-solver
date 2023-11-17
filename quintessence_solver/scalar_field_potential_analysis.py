# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:51:29 2023

@author: Sebastiano
"""

import numpy as np
import matplotlib.pyplot as pl

import quintessence_solver 

import sys

sys.path.append("../data_modules")
import simulation_parameters as params 
import cosmological_functions as cosm_func


sys.path.append("../utility_modules")
import plotting_functions as mypl
import import_export as myie


#%%

"""If compute_approximations=False the program do not compute also the quantities using
the true step equation of state."""
compute_approximations=True

var_par="trans_z" # can be = 'w_i';'w_f';'trans_steepness';'trans_z';
init_value=-0.5
final_value=-0.9
n_samples=3



"""Print out the other parameters to double check their values."""
for key,value in params.dark_energy_eos_params.items():
    print(f"{key}: {value}")
    
#%%

"""Checks: if the number of samples is choosen as zero, we solve 
for only one var_par value """
if n_samples>0:
    var_par_values=np.linspace(init_value,final_value,n_samples)
    var_par_values=[0.3,1.2,3,5]
    # var_par_values=[0.1,1,2,10]
    n_samples=len(var_par_values)
else:
    raise Exception("The number of samples must be greater than zero")

"""We save the original value of the varied parameter, then
we change the parameter values in the params namespace so
that all moduels notify the change."""
exec("reset_param_value=params."+var_par)#Save the original inital value to reset it

#%%
params_names={"w_i":"w_i",
                   "w_f":"w_f",
                   'trans_z':"z_t",
                   'trans_steepness':"\Gamma"}

phi_tilde_numerical_a=[]
approx_phi_tilde_numerical_a=[]
potential_tilde_numerical_a=[]
approx_potential_tilde_numerical_a=[]
potential_tilde_numerical_phi=[]
approx_potential_tilde_numerical_phi=[]

scale_param_values=np.linspace(params.a_min,params.a_max,500)
dark_energy_eos=[]
approx_dark_energy_eos=[]

legend=[]
for i in range(len(var_par_values)):
    exec("params."+var_par+"="+str(var_par_values[i]))#Update the var_par value across the modules.
    legend.append("$"+params_names[var_par]+"$="+str(round(var_par_values[i],3)))
                  
    quintessence_sol=quintessence_solver.solve(compute_approximations)
    
    """Build some intersting quantities matrices to plot: H(a),w(a),dark_density_evolution(a),omega_m(a)..."""
    phi_tilde_numerical_a.append(np.array(quintessence_sol.phi_tilde_numerical_a))
    approx_phi_tilde_numerical_a.append(np.array(quintessence_sol.approx_phi_tilde_numerical_a))
    potential_tilde_numerical_a.append(np.array(quintessence_sol.potential_tilde_numerical_a))
    approx_potential_tilde_numerical_a.append(np.array(quintessence_sol.approx_potential_tilde_numerical_a))
    potential_tilde_numerical_phi.append(np.array(quintessence_sol.potential_tilde_numerical_phi))
    approx_potential_tilde_numerical_phi.append(np.array(quintessence_sol.approx_potential_tilde_numerical_phi))
    
    dark_energy_eos.append(np.array([scale_param_values,params.de_eos_a(scale_param_values)]))
    if compute_approximations:
        step_fun=np.vectorize(cosm_func.step_eos)
        approx_dark_energy_eos.append(np.array([scale_param_values,step_fun(scale_param_values)]))
    
    """Print a table with some informations to keep track where is the cycle at."""
    labels=["Var_par:"+var_par,"t_0","q_0"]

    values=[round(var_par_values[i],3),"#","#"]
    if i==0:
        print("*" * 50)
        print("{:<27} {:<12} {:<12}".format(*labels))
        print("*" * 50)
        print("{:<27} {:<12} {:<12}".format(*values))
        print("*" * 50)
    else:
        print("{:<27} {:<12} {:<12}".format(*values))
        print("*" * 50)
        
    
    
exec("params."+var_par+"=reset_param_value")#Reset the original value 

#%%
"""SAVE DATA AND PLOTS """
save=False#Save the plots?

this_run_specifier_0="quintessence_model"
this_run_specifier_1=params.selected_de_eos
this_run_specifier_2=var_par

save_data_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/"+this_run_specifier_2+"/"
save_plot_to_path="../data/"+this_run_specifier_0+"/"+this_run_specifier_1+"/plots/"+this_run_specifier_2+"/"


if True:
    myie.save_to_txt_multicolumn(phi_tilde_numerical_a,save_data_to_path+"phi_tilde_a",var_par,var_par_values)
    myie.save_to_txt_multicolumn(approx_phi_tilde_numerical_a,save_data_to_path+"approx_phi_tilde_a",var_par,var_par_values)
    myie.save_to_txt_multicolumn(potential_tilde_numerical_a,save_data_to_path+"potential_tilde_a",var_par,var_par_values)
    myie.save_to_txt_multicolumn(approx_potential_tilde_numerical_a,save_data_to_path+"approx_potential_tilde_a",var_par,var_par_values)
    myie.save_to_txt_multicolumn(potential_tilde_numerical_phi,save_data_to_path+"potential_tilde_phi",var_par,var_par_values)
    myie.save_to_txt_multicolumn(approx_potential_tilde_numerical_phi,save_data_to_path+"approx_potential_tilde_phi",var_par,var_par_values)
    myie.save_to_txt_multicolumn(dark_energy_eos,save_data_to_path+"eos",var_par,var_par_values)
    myie.save_to_txt_multicolumn(approx_dark_energy_eos,save_data_to_path+"approx_eos",var_par,var_par_values)


#%%



"""DARK ENERGY EOS """
mypl.plot(dark_energy_eos,
            xlabel=r"$a$",
            ylabel=r"$w$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="de_eos",
            xscale="linear",yscale="linear",
            xlim=None,ylim=None, dotted=False)


"""SCALAR FIELD TILDE of a """
mypl.plot(phi_tilde_numerical_a,
            xlabel=r"$a$",
            ylabel=r"$\tilde{\phi} (a)$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="scalar_field_tilde_a",
            xscale="linear",yscale="linear",
            xlim=None,ylim=None, dotted=False)


    
"""POTENTIAL TILDE of phi"""
mypl.plot(potential_tilde_numerical_phi,
            xlabel=r"$\tilde{\phi}$",
            ylabel=r"$\tilde{V}(\tilde{\phi})$",
            title="",
            legend=legend,
            save_plot_dir=save_plot_to_path,
            save=save,name="potential_tilde_phi",xscale="linear",yscale="log",
            xlim=None,ylim=(0,1e9), dotted=False)



"""POTENTIAL TILDE of a"""
mypl.plot(potential_tilde_numerical_a,
            xlabel=r"$a$",
            ylabel=r"$\tilde{V}(a)$",
            title="",
            legend=legend,
            # func_to_compare=lambda a :omega_dark_now/(a**3),
            save_plot_dir=save_plot_to_path,
            save=save,name="potential_tilde_a",
            xscale="linear",yscale="log",
            xlim=None,ylim=(0.8,1e3), dotted=False)



if compute_approximations:
    approx_legend=legend[:]
    if var_par=="trans_steepness":
        approx_legend.append("Approx.")
    else:
        for item in legend:
            approx_legend.append("Approx. "+item)
    
    aux0=phi_tilde_numerical_a[:]
    for item in approx_phi_tilde_numerical_a:
        aux0.append(item)
    """approximated SCALAR FIELD TILDE of a """
    mypl.plot(aux0,
                xlabel=r"$a$",
                ylabel=r"$\tilde{\phi} (a)$",
                title="",ncol=2,
                legend=approx_legend,
                save_plot_dir=save_plot_to_path,location="upper left",
                save=save,name="approx_scalar_field_tilde_a",
                xscale="linear",yscale="linear",
                xlim=None,ylim=None, dotted=False)
     
    aux1=potential_tilde_numerical_phi[:]
    for item in approx_potential_tilde_numerical_phi:
          aux1.append(item)
          
    """approximated POTENTIAL TILDE of phi"""
    mypl.plot(aux1,
                xlabel=r"$\tilde{\phi}$",
                ylabel=r"$\tilde{V}(\tilde{\phi})$",
                title="",
                ncol=2,
                legend=approx_legend,
                save_plot_dir=save_plot_to_path,
                save=save,name="approx_potential_tilde_phi",xscale="linear",yscale="log",
                xlim=None,ylim=(0,1e9), dotted=False)

     
        
