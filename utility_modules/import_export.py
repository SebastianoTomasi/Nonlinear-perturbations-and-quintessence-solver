# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:48 2023

@author: sebas
"""
import sys

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import numpy as np

sys.path.append("../data_modules")
import simulation_parameters as params

#%%


format_string='{:.9e}'
header_delimiter="*********************************\n"
data_delimiter=  "#################################\n"




def insert_header(file,var_par=None,var_par_values=None):
    file.write("Run parameters:\n")
    for key, value in params.pseudo_newtonian_perturbations_run_params.items():
        file.write(f"{key}={value}\n")
    file.write("\n")
    
    file.write("Cosmological parameters:\n")
    for key, value in params.cosmological_params.items():
        file.write(f"{key}={value}\n")
    file.write("\n")
    
    file.write("Dark energy eos parameters:\n")
    
    if var_par!=None:
        var_par_str = ','.join(map(str, var_par_values))
    else:
        var_par_str=""
    for key, value in params.dark_energy_eos_params.items():
        if key==var_par:
            file.write(f"{key}={var_par_str}\n")
        else:
            file.write(f"{key}={value}\n")
    file.write("\n")
    
    file.write("Precision parameters:\n")
    for key, value in params.precision_params.items():
        file.write(f"{key}={value:.1e}\n")
    file.write("\n")
    file.write(header_delimiter)

def save_to_txt_twocolumns(data,path):
    if len(data)!=2:
        raise Exception("Data must has two columns: data=[[list_1],[list_2]]")
    
    lenght=len(data[0])
    if lenght!=len(data[1]):
        raise Exception("The dimension of the two columns must be equal!")
        
    with open(path+".txt","w") as file:
        insert_header(file)
        for i in range(lenght):
            x=format_string.format(data[0][i])
            y=format_string.format(data[1][i])
            file.write(x+"\t"+y+"\n")


def import_sigma8(path):
    try:
        with open(path+".dat", 'r') as file:
            sigma8 = float(file.readline())
            return sigma8
    except:
        raise Exception(f"{path}: File not found")


def import_class_pk(relative_path):
    start_reading = False
    data = [[], []]

    with open(relative_path+".dat", 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip lines starting with '#'
            elif not start_reading:
                start_reading = True
            else:
                line = line.split()
                data[0].append(float(line[0]))
                data[1].append(float(line[1]))
    return data



def import_from_txt_twocolumns(path):
    line="line"
    start_reading=False
    data=[[],[]]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if line==header_delimiter:
                start_reading=True
            elif line=="":
                return data
            elif start_reading:
                line=line.split()
                data[0].append(float(line[0]))
                data[1].append(float(line[1]))



def save_to_txt_multicolumn(data,path,var_par=None,var_par_values=None):
    with open(path+".txt","w") as file:
        if isinstance(var_par, str) and isinstance(var_par_values, (np.ndarray,tuple,list)):
            insert_header(file,var_par,var_par_values)
        for i in range(len(data)):
            length=len(data[i][0])
            for j in range(length+1):
                if j==length:
                    file.write(data_delimiter)
                else:
                    x=format_string.format(data[i][0][j])
                    y=format_string.format(data[i][1][j])
                    file.write(x+"\t"+y+"\n")

def import_from_txt_multicolumn(path):
    start_reading=False          
    line="line"
    data=[]
    sub_data=[[],[]]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if start_reading:
                if line=="":
                    return data
                elif line==data_delimiter:
                    data.append(np.array(sub_data))
                    sub_data=[[],[]]
                else:
                    line=line.split()
                    sub_data[0].append(np.array(float(line[0])))
                    sub_data[1].append(np.array(float(line[1])))
            if line==header_delimiter:
                start_reading=True

def import_varied_parameter_values(path,var_par):
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            if var_par in line and "varied_parameter" not in line:
                aux=eval(line[len(var_par)+1:-1])
                if isinstance(aux, (float,int)):
                    return [aux]
                else:
                    return aux

def save_to_txt_timedata(time_data, path="/run/media/sebastianotomasi/SebaSSD/Synced/OneDrive/python_code/time_data_test",var_par=None,var_par_values=None):

    tot_len = len(time_data)
    sub_len = len(time_data[0])
    str_lengths=list(map(len,time_data[0]))
    with open(path + ".txt", "w") as file:
        insert_header(file,var_par,var_par_values)
        for j in range(0,sub_len):
            x = time_data[0][j]
            if j != sub_len-1:  # Use sub_len - 1 to check for the last element
                file.write(str(x) + "\t")
            else:
                file.write(str(x) + "\n")
        for i in range(sub_len):
            for j in range(1,tot_len):
                x = str(time_data[j][i])  # Convert to string
                if j != tot_len-1:  # Use tot_len - 1 to check for the last element
                    space=" "*(str_lengths[j-1]-len(x))
                    file.write(x + space+"\t")
                else:
                    file.write(x + "\n")
                    
def save_to_txt_sigma8(sigma8_values, path="test",var_par=None,var_par_values=None):

    with open(path + ".txt", "w") as file:
        row=var_par+"\t"
        for var_par_value in var_par_values:
            row+="$"+str(var_par_value)+"$ &"+"\t"
        file.write(row + "\n")
        
        row="sigma8\t"
        for sigma8 in sigma8_values:
            row+="$"+str(sigma8)+"$ &"+"\t"
        file.write(row + "\n")



            
"""Import the header to save it in a separated txt in the plots folder."""
def import_header(path):
    header=[]
    with open(path+".txt","r") as file:
        while True:
            line=file.readline()
            header.append(line)
            if line==header_delimiter:
                return header
            
def save_header(header,path):
    with open(path+".txt","w") as file:
        for line in header:
            file.write(line)


"""Create folder"""
def create_dir(path,name):
    paths = os.path.join(path, name) 
    print(paths)
    try:
        os.mkdir(paths)
        print(f"Directory '{name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{name}' already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")
        
"""Generates a structered tree of folders to store the data and the plots."""
def generate_data_folders():
    name_specifiers_0=["nonlinear_perturbations","linear_perturbations","quintessence_model","mass_function"]
    name_specifiers_1=["unperturbed_de","perturbed_de","LCDM","EDS"]
    name_specifiers_2=["de_eos_"+str(i) for i in range(1,7)]
    name_specifiers_3=['w_i','w_f','trans_steepness','trans_z']
    
    try:
        os.mkdir("../data")
    except:
        pass
    
    for spec_0 in name_specifiers_0:
        create_dir("../data/", spec_0)
        if spec_0=="quintessence_model":
            for spec_2 in name_specifiers_2:
                create_dir("../data/"+spec_0+"/", spec_2)
                create_dir("../data/"+spec_0+"/"+"/"+spec_2, "plots")
                for spec_3 in name_specifiers_3:
                    create_dir("../data/"+spec_0+"/"+spec_2, spec_3)
                    create_dir("../data/"+spec_0+"/"+spec_2+"/"+"plots", spec_3)
            
            
        else:
            for spec_1 in name_specifiers_1:
                if spec_1=="LCDM":
                    create_dir("../data/"+spec_0, "LCDM")
                elif spec_1=="EDS":
                    create_dir("../data/"+spec_0, "EDS")
                elif spec_1=="unperturbed_de" or spec_1=="perturbed_de":
                    create_dir("../data/"+spec_0, spec_1)
                    for spec_2 in name_specifiers_2:
                        create_dir("../data/"+spec_0+"/"+spec_1, spec_2)
                        create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2, "plots")
                        for spec_3 in name_specifiers_3:
                            create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2, spec_3)
                            create_dir("../data/"+spec_0+"/"+spec_1+"/"+spec_2+"/"+"plots", spec_3)
                            


    
                    
    
# file=open("test.txt","w")
# insert_header(file,"w_i",[1,2,3])
# file.close()

# test_header=import_header("test")
# save_header(test_header,"saved_header")

# test_data=[[1,2,3,4,5,6,7,8,9,10],[1,4,9,16,25,36,49,64,81,100]]
# test_name="test"
# save_to_txt_twocolumns(test_data,test_name)
# import_twocolumn_test=import_from_txt_twocolumns(test_name)

# multicol_test_data=[test_data,test_data,test_data,test_data]
# save_to_txt_multicolumn(multicol_test_data,test_name)

# import_multicolumn_test=import_from_txt_multicolumn(test_name)
