# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:06:24 2023

@author: Sebastiano Tomasi
"""


import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import scipy as sp

import numpy as np
import statsmodels.api as sm

import mass_function_aux as aux
import sys
sys.path.append("../utility_modules")
import import_export as myie
import numerical_methods as mynm
import cosmological_functions as cosm_func
import plotting_functions as mypl

sys.path.append("../data_modules")
import simulation_parameters as params




#%%

save=True#Save the plots


this_run_specifier_0= "nonlinear_perturbations"
save_plot_to_path="../data/mass_function/perturbed_de/"+params.this_run_specifier_2+"/plots/"+params.this_run_specifier_3+"/"

get_deltac_from = "../data"+"/"+this_run_specifier_0+"/" + params.this_run_specifier_1+"/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/delta_c"



get_hmf_cs2_equal_0_path="../data/mass_function/perturbed_de/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/"
get_hmf_cs2_equal_1_path="../data/mass_function/unperturbed_de/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/"

masses=mynm.logspace(params.min_mass,params.max_mass,params.number_of_masses)
#%%
var_par_values = myie.import_varied_parameter_values(get_deltac_from, params.this_run_specifier_3)
len_var_par_values=len(var_par_values)

legend=[]
params_names = {"w_i": "w_i",
                "w_f": "w_f",
                "trans_steepness": "\Gamma",
                "trans_z": "z_t",
                "de_eos_a": params.this_run_specifier_2}
st_hmf_scaled=[]
st_hmf_scaled_percent=[]#model=lcdm(1+%)
ps_scaling_factor_percentage=[]
st_scaling_factor_percentage=[]

scaled_to_st_cs_equal_0_ratio=[]

lcdm=[]
for redshift_index in range(1,len(params.z)+1):

    st_cs_equal_1_numerical=myie.import_from_txt_multicolumn(get_hmf_cs2_equal_1_path+str(redshift_index-1)+"_st_hmf")
    st_cs_equal_0_numerical=myie.import_from_txt_multicolumn(get_hmf_cs2_equal_0_path+str(redshift_index-1)+"_st_hmf")
    ps_cs_equal_1_numerical=myie.import_from_txt_multicolumn(get_hmf_cs2_equal_1_path+str(redshift_index-1)+"_ps_hmf")
    ps_cs_equal_0_numerical=myie.import_from_txt_multicolumn(get_hmf_cs2_equal_0_path+str(redshift_index-1)+"_ps_hmf")
    st_lcdm=myie.import_from_txt_twocolumns("../data/mass_function/LCDM/"+str(redshift_index-1)+"_lcdm_st_hmf")
    
    st_hmf_scaled.append(st_lcdm)
    
   
    for i in range(len_var_par_values):
        ps_ratio_cs0_cs1=ps_cs_equal_0_numerical[i][1]/ps_cs_equal_1_numerical[i][1]
        st_ratio_cs0_cs1=st_cs_equal_0_numerical[i][1]/st_cs_equal_1_numerical[i][1]
        
        ps_scaling_factor_percentage.append([mynm.mooving_average(ps_cs_equal_0_numerical[i][0],5),mynm.mooving_average((ps_ratio_cs0_cs1-1)*100,5)])
        st_scaling_factor_percentage.append([mynm.mooving_average(st_cs_equal_0_numerical[i][0],5),mynm.mooving_average((st_ratio_cs0_cs1-1)*100,5)])
        
        scaled_to_st_cs_equal_0_ratio.append([st_cs_equal_1_numerical[i][0],(ps_ratio_cs0_cs1/st_ratio_cs0_cs1-1)*100])
        
        st_hmf_scaled.append([st_cs_equal_1_numerical[i][0],st_cs_equal_1_numerical[i][1]*ps_ratio_cs0_cs1])
        st_hmf_scaled_percent.append([st_hmf_scaled[-1][0],(st_hmf_scaled[-1][1]/st_lcdm[1]-1)*100])                    
        legend.append("$"+params_names[params.this_run_specifier_3]+"$="+str(var_par_values[i])+"   z="+str(params.z[redshift_index-1]))

for i in range(1,len(params.z)+1):
    legend.append(r"$\Lambda$CDM   z="+str(params.z[redshift_index-1]))
    
mypl.plot(f=ps_scaling_factor_percentage,
            xlabel=r"Mass  $[M_\odot h^{-1}]$",
            ylabel=r"Difference $[\%]$",
            title=None,
            xscale="log", 
            # yscale="log",
            ncol=2,
            # dotted=True,
            # ylim=(0.7,1.3),
            xlim=(1e10,1e13),
            # y_ticks=[0.7,0.8,0.9,1,1.1,1.2],
            legend=[legend,None],save=save,
            save_plot_dir=save_plot_to_path,name="ps_scaling_factor_percentage")

mypl.plot(f=st_scaling_factor_percentage,
            xlabel=r"Mass  $[M_\odot h^{-1}]$",
            ylabel=r"Difference $[\%]$",
            title=None,
            xscale="log", 
            # yscale="log",
            ncol=2,
            # dotted=True,
            # ylim=(0.7,1.3),
            xlim=(1e10,1e13),
            # y_ticks=[0.7,0.8,0.9,1,1.1,1.2],
            legend=[legend,None],save=save,
            save_plot_dir=save_plot_to_path,name="st_scaling_factor_percentage")

mypl.plot(f=scaled_to_st_cs_equal_0_ratio,
            xlabel=r"Mass  $[M_\odot h^{-1}]$",
            ylabel=r"Difference $[\%]$",
            title=None,
            xscale="log", 
            # yscale="log",
            ncol=2,
            # dotted=True,
            ylim=(-4,0.4),
            # xlim=(1e10,1e13),
            # y_ticks=[0.7,0.8,0.9,1,1.1,1.2],
            legend=[legend,None],save=save,
            save_plot_dir=save_plot_to_path,name="scaled_to_st_percentage")

# mypl.plot(f=[st_hmf_scaled,st_hmf_scaled_percent],
#             xlabel=["",r"Mass  $[M_\odot h^{-1}]$",],
#             ylabel=[r"Scaled ST hmf",r"Difference  $[\%]$"],
#             title=None,
#             xscale=["log","log"], 
#             yscale=["log","linear"],
#             ncol=2,
#             # dotted=True,
#             legend=[legend,None],
#             save=save,
#             save_plot_dir=save_plot_to_path,name="scaled_hmf")


        
