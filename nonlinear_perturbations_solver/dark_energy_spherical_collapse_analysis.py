# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:29:28 2023

@author: sebas
"""


import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate
import scipy.integrate

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import sys
sys.path.append("../data_modules")
import simulation_parameters as params
import cosmological_functions as cosm_func
import data_classes as dtc

sys.path.append("../utility_modules")
import plotting_functions as mypl
import import_export as myie

sys.path.append("../friedmann_solver")
import friedmann_solver as fr_sol


import pseudo_newtonian_perturbations as psp
# import clustering_de_spherical_collapse as psp


import time

#%%

"""
This module computes the nonlinear perturbation_results for the dark energy 
model and parameters specified in the parameters module. 
The variable params.varied_parameter is the parameter of the eos we want to vary. We
vary it in the range [params.init_value,params.final_value] with params.n_samples points.
The specifiers must be correctly named according to the folder structure of 
the method import_export.generate_data_folders().
If one want to switch from unperturbed dark energy to perturbed, can change the value
of the boolean psp.does_de_cluster to True."""


"""Print out the other parameters to double check their values."""

# print("COSMOLOGICAL PARAMETS:")
# mypl.print_dict(params.cosmological_params)
# print("\n")

print("DARK ENERGY EOS PARAMETERS:")
mypl.print_dict(params.dark_energy_eos_params)
print("\n")

# print("PRECISION PARAMETERS:")
# mypl.print_dict(params.precision_params)
# print("\n")

print("RUN PARAMETERS:")
mypl.print_dict(params.pseudo_newtonian_perturbations_run_params)


#%%


def solve(only_linear=False):
    start_time = time.time()
    """Check if we are in some particular universes."""
    wCDM=params.selected_de_eos=="wcdm"
    EDS=params.omega_matter_now==0 and params.omega_rad_now==0
    
    if  wCDM and params.varied_parameter!="w":
        raise Exception("If you select wcdm you must vary w")
        
    """Check if the number of samples is positive"""
    if params.n_samples>0:
        """Check if the selected model is the wcdm"""
        LCDM = wCDM and params.n_samples==1  and params.init_value==-1
        if params.varied_par_values==None:
            varied_par_values=np.linspace(params.init_value,params.final_value,params.n_samples)
        elif isinstance(params.varied_par_values,list):
            varied_par_values=params.varied_par_values
            params.n_samples=len(varied_par_values)
        else:
            raise Exception("Somthing is wrong in the input parameters, check if varied_par_values is a list varied_par_values={params.varied_par_values}")
        print(params.varied_parameter+" choosen values"+"="+str(varied_par_values)+"\n")
    else:
        raise Exception("The number of samples must be greater or equal to zero")
    
    #%%
    
    """Construct the path where to save the data."""
    this_run_specifier_0="nonlinear_perturbations"
    
    
    """Check if de can cluster or not"""
    if LCDM:
        this_run_specifier_1=""
        this_run_specifier_2="LCDM"
        this_run_specifier_3=""
    elif EDS:
        this_run_specifier_1=""
        this_run_specifier_2="EDS"
        this_run_specifier_3=""
    else:
        if params.does_de_cluster:
            this_run_specifier_1="perturbed_de"
        else:
            this_run_specifier_1="unperturbed_de"
        this_run_specifier_2=params.selected_de_eos
        this_run_specifier_3=params.varied_parameter
        
        
    
    
    
    
    save_data_to_path=os.path.join("../data",this_run_specifier_0,this_run_specifier_1,this_run_specifier_2,this_run_specifier_3)
    save_plot_to_path=os.path.join("../data",this_run_specifier_0,this_run_specifier_1,this_run_specifier_2,"plots",this_run_specifier_3)
    
    #%%
    """Actual cycle on the parameter values."""
    
    params_names={"w_i":"w_i",#Used to build the legend
                "w_f":"w_f",
                "trans_steepness":"\Gamma",
                "trans_z":"z_t",
                "de_eos_a":params.selected_de_eos}
    
    legend=[]
    time_data=[["Var_par: "+params.varied_parameter,"t_0","q_0"],varied_par_values,[],[]]
    
    """Classes to store results"""
    perturbation_results=dtc.perturbation_results()
    background_results=dtc.friedmann_sol()

    for i in range(len(varied_par_values)):
        exec("params."+params.varied_parameter+"="+str(varied_par_values[i]))#Update the params.varied_parameter value across the modules.
        """Construct the legend"""
        
        if LCDM:
            legend.append("$\Lambda$CDM")
        elif wCDM:
            legend.append("w="+str(round(varied_par_values[i],3)))
        else:
            legend.append("$"+params_names[params.varied_parameter]+"$="+str(round(varied_par_values[i],3)))
            
        
        """Call the friedmann solver.
        If params.params.time_domain=True then it will print out the deceleration
        parameter and the age of the universe for the model considered."""      
        friedmann_sol=fr_sol.solve(time_domain=params.time_domain)
        background_results.append_instance(friedmann_sol)
        
        """Call the spherical collapse solver. It may be for backround or clustering dark energy.""" 
        nonlin_pertrubations=psp.solve(friedmann_sol,only_linear)
        perturbation_results.linear_matter_density_contrast_a.append(nonlin_pertrubations[2][0])
        perturbation_results.linear_de_density_contrast_a.append(nonlin_pertrubations[2][1])
        if not only_linear:
            perturbation_results.linear_matter_density_contrasts_at_collapse_z.append(nonlin_pertrubations[0])
            perturbation_results.virialization_overdensities_star_z.append(nonlin_pertrubations[1])

        
        
        if params.time_domain:
            time_data[2].append(round(friedmann_sol.universe_age,3))
            time_data[3].append(round(friedmann_sol.deceleration_parameter,3))
            
        else:
            time_data[2].append("#")
            time_data[3].append("#")
    

        
    # Transpose all elements of time_data except the first one
    transposed_arr = [time_data[0]] + list(map(list, zip(*time_data[1:])))
    
    # Find the maximum length of string in each column
    max_lengths = [max(len('{:.3f}'.format(elem)) if isinstance(elem, (int, float)) else len(str(elem)) for elem in col) for col in zip(*transposed_arr)]
    
    # Print each row of the array
    for row in transposed_arr:
        print('*' * (sum(max_lengths) + len(max_lengths) * 3 + 4))  # Print line of asterisks
        print('* ' + ' * '.join(['{:.3f}'.format(elem).ljust(max_lengths[i]) if isinstance(elem, (int, float)) else str(elem).ljust(max_lengths[i]) for i, elem in enumerate(row)]) + ' *')  # Print row with numbers rounded to 3 decimal places
    print('*' * (sum(max_lengths) + len(max_lengths) * 3 + 4))  # Print final line of asterisks
    
    
    exec("params."+params.varied_parameter+"=params.dark_energy_eos_params['"+params.varied_parameter+"']")#Reset the original value 
    #%%
    
    """Save all data to .txt files in the correct folders"""
    if params.save_data:
        """Save linear perturbations"""
        myie.save_to_txt_multicolumn(perturbation_results.linear_matter_density_contrast_a, os.path.join(save_data_to_path,"growth_factor"),params.varied_parameter,varied_par_values)
        myie.save_to_txt_multicolumn(perturbation_results.linear_de_density_contrast_a, os.path.join(save_data_to_path,"dark_growth_factor"),params.varied_parameter,varied_par_values)
        if not only_linear:
            """Save nonlinear perturbations"""
            myie.save_to_txt_multicolumn(perturbation_results.linear_matter_density_contrasts_at_collapse_z, os.path.join(save_data_to_path,"delta_c"),params.varied_parameter,varied_par_values)
            myie.save_to_txt_multicolumn(perturbation_results.virialization_overdensities_star_z, os.path.join(save_data_to_path,"zeta_vir_star"),params.varied_parameter,varied_par_values)
            """Save background"""
            myie.save_to_txt_multicolumn(background_results.effective_eos_numerical_a, os.path.join(save_data_to_path,"effective_eos"),params.varied_parameter,varied_par_values)
            myie.save_to_txt_multicolumn(background_results.de_eos_numerical_a, os.path.join(save_data_to_path,"dark_energy_eos"),params.varied_parameter,varied_par_values)
            myie.save_to_txt_multicolumn(background_results.matter_density_parameter_numerical_a, os.path.join(save_data_to_path,"omega_matter"),params.varied_parameter,varied_par_values)
        
        """Save time data"""
        myie.save_to_txt_timedata(time_data, os.path.join(save_data_to_path,"age_deceleration"),params.varied_parameter,varied_par_values)

    
    perturbation_results.legend=legend
    background_results.legend=legend
    print("Time taken to complete the execution: {:.2f} seconds".format(time.time() - start_time))
    return [perturbation_results,background_results]