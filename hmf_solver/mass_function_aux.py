# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:09:55 2023

@author: sebas
"""

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

def print_info_class(use_class):
    if use_class:
        print("Using CLASS spectrum.")
    elif not use_class:
        print("Using transfered spectrum.")

def krange_is_ok(class_k_values,min_k,max_k):
    if max_k>max(class_k_values) or min_k<min(class_k_values):
        
        return False
    else:
        return True


def check_if_parameters_match(path_to_nonlinear,path_to_class):
    parameter_names = {"w_i": "wi_fld",
                    "w_f": "wf_fld",
                    "trans_steepness": "gamma_fld",
                    "trans_z": "zt_fld"}

    nonlinear_perturbations_params={"w_i":0,
                                    "w_f": 0,
                                    "trans_steepness": 0,
                                    "trans_z": 0}
    with open(path_to_nonlinear+".txt","r") as file:
        for line in file:
            for param_name,value in nonlinear_perturbations_params.items():
                if param_name in line and "varied_parameter" not in line:
                    nonlinear_perturbations_params[param_name]=eval(line.replace("="," ").split()[-1])

                    
    for param_name,value in nonlinear_perturbations_params.items():
        if isinstance(value, tuple):
            iterations=len(value)
            varied_param_name=param_name
        else:
            iterations=1
    for number in range(iterations):
        with open(path_to_class+str(number)+"_parameters.ini","r") as file:
            for line in file:
                for param_name,class_param_name in parameter_names.items():
                    if class_param_name in line:
                        class_param_value=float(line.split()[-1])
                        if varied_param_name==param_name:
                            if class_param_value!=nonlinear_perturbations_params[param_name][number]:
                                print(f"Parameter={param_name}\t nonlinear_pert:{nonlinear_perturbations_params[param_name][number]}\t class:{class_param_value}")
                                return False
                        else:
                            if class_param_value!=nonlinear_perturbations_params[param_name]:
                                print(f"Parameter={param_name}\t nonlinear_pert:{nonlinear_perturbations_params[param_name]}\t class:{class_param_value}")
                                return False
    return True
        




        
