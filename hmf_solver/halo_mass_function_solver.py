# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:42:03 2023

@author: Sebastiano
"""


import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import scipy as sp
# import scipy.interpolate
import numpy as np

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

if  params.this_run_specifier_1=="perturbed_de":
    this_run_specifier_4="cs2_equal_0"
elif params.this_run_specifier_1=="unperturbed_de":
    this_run_specifier_4="cs2_equal_1"

this_run_specifier_0= "nonlinear_perturbations"
save_data_to_path="../data/mass_function"+"/"+params.this_run_specifier_1+"/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/"
save_plot_to_path="../data/mass_function"+"/"+params.this_run_specifier_1+"/"+params.this_run_specifier_2+"/plots/"+params.this_run_specifier_3+"/"

get_deltac_from = "../data"+"/"+this_run_specifier_0+"/" + params.this_run_specifier_1+"/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/delta_c"
get_growth_from="../data"+"/"+"nonlinear_perturbations"+"/" +params.this_run_specifier_1+"/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/growth_factor"



get_growth_lcdm="../data/nonlinear_perturbations/LCDM/growth_factor"
get_deltac_lcdm="../data/nonlinear_perturbations/LCDM/delta_c"
class_data_lcdm="../data/class/LCDM/"

k=mynm.logspace(params.min_k,params.max_k,5e2)
masses=mynm.logspace(params.min_mass,params.max_mass,params.number_of_masses)

#%%

"""Define wich window function to use"""
if params.window_function=="TH":
    def W(k,M):#window function
        return cosm_func.TH_window_function(k, M)
    def dW2(k,M):#window function squared w.r.t k
        return cosm_func.TH_window_function2_derivative(k, M)
elif params.window_function=="GAUS":
    def W(k,M):#window function
        return cosm_func.GAUS_window_function(k, M)
    def dW2(k,M):#window function squared w.r.t k
        return cosm_func.GAUS_window_function2_derivative(k, M)
"""Define which hmf compute"""
if params.fitting_func=="PS":
    f=cosm_func.f_ps
elif params.fitting_func=="ST":
    f=cosm_func.f_st
else:
    raise Exception(f"{params.fitting_func} not recognized.")


def pk_normalize(not_normalized_power_spectrum,sigma8):
    """Function that normalize the power spectrum such that sigma(mass8)=sigma8"""
    spectrum_normaliz=1
    def normalization_integrand(k):
        return 1/(2*np.pi**2)*k**2*not_normalized_power_spectrum(k,spectrum_normaliz)*W(k,params.mass8)**2
    spectrum_normaliz=sigma8**2/mynm.integrate(normalization_integrand, params.min_k, params.max_k,atol=params.normalization_integrand_atol,rtol=params.normalization_integrand_atol)[1][-1]
    print("Normalization = "+f"{spectrum_normaliz:.2e}")
    def pk_normalized(k):
        """Normalized power spectrum"""
        return not_normalized_power_spectrum(k,spectrum_normaliz)
    return pk_normalized

#%%
"""Compute sigma(z) and the logarithmic derivative of log(sigma) for LCDM"""


try:
    """Import the growth factor for LCDM"""
    numerical_growth_factor_lcdm_a=myie.import_from_txt_multicolumn(get_growth_lcdm)
    
    """Import the density contrast at collapse for LCDM."""
    numerical_delta_c_lcdm = myie.import_from_txt_multicolumn(get_deltac_lcdm)    
except:
    raise Exception("LCDM data not found.")
"""Normalize the growth factor such that D(z=0)=1"""
normalized_numerical_growth_factor_lcdm_z=[cosm_func.z_a(np.array(numerical_growth_factor_lcdm_a[0][0])),\
                                           np.array(numerical_growth_factor_lcdm_a[0][1])/\
                                               numerical_growth_factor_lcdm_a[0][1][-1]]
growth_factor_lcdm_z=sp.interpolate.interp1d(normalized_numerical_growth_factor_lcdm_z[0],
                                             normalized_numerical_growth_factor_lcdm_z[1],
                                        fill_value="extrapolate", assume_sorted=False)

delta_c_lcdm=sp.interpolate.interp1d(np.array(numerical_delta_c_lcdm[0][0]),
                                    numerical_delta_c_lcdm[0][1],
                                    fill_value="extrapolate", assume_sorted=False)

""""""

pk_lcdm=[]
pk_nl_lcdm=[]
hmf_lcdm=[]
for redshift_index in range(1,len(params.z)+1):
    class_pk_lcdm_numerical=myie.import_class_pk(class_data_lcdm+"_z"+str(redshift_index)+"_pk")
    
    """Check if the ranges for k nicely overlap. """
    if not aux.krange_is_ok(class_pk_lcdm_numerical[0], params.min_k, params.max_k):
        raise Exception(f"The provided k range [{params.min_k}, {params.max_k}] is not compatible with the CLASS data.\
                        Please adjust k_min and k_max to be within the valid range\
                            [{class_pk_lcdm_numerical[0][0]}, {class_pk_lcdm_numerical[0][-1]}].")
                            
    nonlinear_power_spectrum_lcdm_numerical=myie.import_class_pk(class_data_lcdm+"_z"+str(redshift_index)+"_pk_nl")
    nonlinear_power_spectrum_lcdm=sp.interpolate.interp1d(nonlinear_power_spectrum_lcdm_numerical[0], nonlinear_power_spectrum_lcdm_numerical[1],
                                        fill_value="extrapolate", assume_sorted=True)
    class_pk_lcdm=sp.interpolate.interp1d(class_pk_lcdm_numerical[0], class_pk_lcdm_numerical[1],
                                        fill_value="extrapolate", assume_sorted=True)


    """Compute the variance of the filtered field"""

    numerical_sigma_M=[]
    for M in masses:
        def sigma_integrand(k):
            return 1/(2*np.pi**2)*k**2*class_pk_lcdm(k)*W(k,M)**2
        numerical_sigma_M.append(np.sqrt(mynm.integrate(sigma_integrand,
                                                        params.min_k, params.max_k,
                                                        atol=params.mass_variance_filtered_spectrum_atol,
                                                        rtol=params.mass_variance_filtered_spectrum_rtol)[1][-1]))
        
    sigma_lcdm=sp.interpolate.interp1d(masses, numerical_sigma_M,
                                        fill_value="extrapolate", assume_sorted=True)
    
    """Compute the logarithmic derivative of log(sigma_lcdm)"""
    numerical_log_der_of_log_sigma=[]
    for M in masses:
        R=params.mass_to_radius*M**(1/3)
        def log_der_integrand(k):
            return dW2(k,M)*class_pk_lcdm(k)/k**2
        numerical_log_der_of_log_sigma.append(3/(2*sigma_lcdm(M)**2*np.pi**2*R**4)*mynm.integrate(log_der_integrand,
                                                                                                  params.min_k, params.max_k,
                                                                                                  rtol=params.log_der_of_log_sigma_rtol,atol=params.log_der_of_log_sigma_atol)[1][-1])
    log_der_of_log_sigma=sp.interpolate.interp1d(masses, numerical_log_der_of_log_sigma,
                                        fill_value="extrapolate", assume_sorted=True)
    
    
    """Compute the mass function for LCDM"""
    mass_function_lcdm=cosm_func.create_mass_function(f,sigma_lcdm,growth_factor_lcdm_z,delta_c_lcdm,log_der_of_log_sigma,params.rho_m0_Ms_Mpc)
    
    pk_lcdm.append(class_pk_lcdm)
    pk_nl_lcdm.append(nonlinear_power_spectrum_lcdm)
    hmf_lcdm.append(mass_function_lcdm)
    myie.save_to_txt_twocolumns([masses,mass_function_lcdm(masses,params.z[redshift_index-1])],path="../data/mass_function/LCDM/"+str(redshift_index-1)+"_lcdm_"+params.fitting_func.lower()+"_hmf")



mypl.plot([[masses, f(masses,params.z[redshift_index])] for redshift_index,f in enumerate(hmf_lcdm)],
                xlabel=r"Mass  $[M_\odot/h]$",ylabel=r"$dn/d\ln M$  $[h^3\,{\rm Mpc}^{-3}]$",
                xscale="log",
                yscale="log",
                title=r"$\Lambda$CDM Halo mass function at various redshifts.",
                # dotted=True,
                legend=["z=0","z=0.5","z=1"],
                save=False,save_plot_dir=save_plot_to_path,name=params.fitting_func.lower()+"_"+this_run_specifier_4+"_lcdm_hmf"
                )


#%%

"""Compute the mass function for a dark energy model"""

try:
    numerical_delta_c = myie.import_from_txt_multicolumn(get_deltac_from)
    numerical_growth_factors_a = myie.import_from_txt_multicolumn(get_growth_from)
    numerical_growth_factors_z=[]
    """Change the growth factor variable from a to z"""
    for i in range(len(numerical_growth_factors_a)):
        numerical_growth_factors_z.append([cosm_func.z_a(np.array(numerical_growth_factors_a[i][0])),
                                          np.array(numerical_growth_factors_a[i][1])])     
except FileNotFoundError:
    raise FileNotFoundError("Error: File not found. Check if the data is available.")

"""Normalize the growth factors to D+(z=0)=1"""
normalized_numerical_growth_factors_z=[]
for gf in numerical_growth_factors_z:
    normalized_numerical_growth_factors_z.append([gf[0],gf[1]/gf[1][-1]])

"""Import the values of the varied parameter"""
var_par_values = myie.import_varied_parameter_values(get_deltac_from, params.this_run_specifier_3)

"""Set up the legend"""
params_names = {"w_i": "w_i",
                "w_f": "w_f",
                "trans_steepness": "\Gamma",
                "trans_z": "z_t",
                "de_eos_a": params.this_run_specifier_2}

halo_mass_functions=[]
mass_functions_percentage=[]#model=lcdm(1+percentage)

linear_power_spectrums=[]
linear_power_spectrums_percentage=[]

nonlinear_power_spectrums=[]
nonlinear_power_spectrums_percentage=[]#model=lcdm(1+percentage)

sigma8_values=[]

legend=[]

class_data_path_base="../data/class/"+params.this_run_specifier_2+"/"+params.this_run_specifier_3+"/"+this_run_specifier_4+"/"
if not aux.check_if_parameters_match(path_to_nonlinear=get_deltac_from,path_to_class=class_data_path_base):
    raise Exception("The parameters in the CLASS simulation do not match the parameters used for the nonlinear density contrast.")
    
for redshift_index in range(1,len(params.z)+1):
    len_var_par_values=len(var_par_values)
    for i in range(len_var_par_values):
        class_data_path=class_data_path_base+str(i)+"_"
        pk_data_path=class_data_path+"z"+str(redshift_index)+"_pk"
        
        """Check if the var_par_values match with the class simulation ones."""
        
    
        """Interpolate delta_c and the growth factor"""
        delta_c=sp.interpolate.interp1d(numerical_delta_c[i][0],numerical_delta_c[i][1],
                                            fill_value="extrapolate", assume_sorted=True)
        
        growth_factor_z=sp.interpolate.interp1d(normalized_numerical_growth_factors_z[i][0],normalized_numerical_growth_factors_z[i][1],
                                                fill_value="extrapolate", assume_sorted=False)
    
        """Before creating the mass fucntion we need to get the power spectrum and normalize it."""
        linear_power_spectrum_numerical=np.array(myie.import_class_pk(pk_data_path))
        linear_power_spectrum=sp.interpolate.interp1d(linear_power_spectrum_numerical[0], linear_power_spectrum_numerical[1],
                                                fill_value="extrapolate", assume_sorted=True)
        
        """Check if the ranges for k nicely overlap. """
        if not aux.krange_is_ok(linear_power_spectrum_numerical[0], params.min_k, params.max_k):
            raise Exception(f"The provided k range [{params.min_k}, {params.max_k}] is not compatible with the CLASS data.\
                            Please adjust k_min and k_max to be within the valid range\
                                [{linear_power_spectrum_numerical[0][0]}, {linear_power_spectrum_numerical[0][-1]}].")
                            
        """Compute the variance of the filtered field"""
        numerical_sigma_M=[]
        for M in masses:
            def sigma_integrand(k):
                return 1/(2*np.pi**2)*k**2*linear_power_spectrum(k)*W(k,M)**2
            numerical_sigma_M.append(np.sqrt(mynm.integrate(sigma_integrand, params.min_k, params.max_k,
                                                            atol=params.mass_variance_filtered_spectrum_atol,
                                                            rtol=params.mass_variance_filtered_spectrum_rtol)[1][-1]))
    
        sigma=sp.interpolate.interp1d(masses, numerical_sigma_M,
                                            fill_value="extrapolate", assume_sorted=True)
        sigma8=np.round(sigma(params.mass8),3)
        
        if redshift_index==1:
            sigma8_values.append(sigma8)
        
        print("z=",params.z[redshift_index-1]," ",params.this_run_specifier_3,"=",var_par_values[i],"sigma8=",sigma8)
    
    
        """Compute the logarithmic derivative of log(sigma)"""
        numerical_log_der_of_log_sigma=[]
        for M in masses:
            R=params.mass_to_radius*M**(1/3)
            def log_der_integrand(k):
                return dW2(k,M)*linear_power_spectrum(k)/k**2
            numerical_log_der_of_log_sigma.append(3/(2*sigma(M)**2*np.pi**2*R**4)*mynm.integrate(log_der_integrand, 
                                                                                                  params.min_k, params.max_k,
                                                                                                  rtol=params.log_der_of_log_sigma_rtol,
                                                                                                  atol=params.log_der_of_log_sigma_atol)[1][-1])
        log_der_of_log_sigma=sp.interpolate.interp1d(masses, numerical_log_der_of_log_sigma,
                                            fill_value="extrapolate", assume_sorted=True)
        """Create the mass function"""
        mass_function=cosm_func.create_mass_function(f,sigma,growth_factor_z,delta_c,log_der_of_log_sigma,params.rho_m0_Ms_Mpc)
        
        """Build the legend"""        
        legend.append("$"+params_names[params.this_run_specifier_3]+"$="+str(var_par_values[i])+"   z="+str(params.z[redshift_index-1]))
        
        """Store results into arrays"""
        mass_functions_percentage.append([masses,(mass_function(masses,params.z[redshift_index-1])/hmf_lcdm[redshift_index-1](masses,params.z[redshift_index-1])-1)*100])
        halo_mass_functions.append([masses,mass_function(masses,params.z[redshift_index-1])])
        
        
        
        linear_power_spectrums.append([k,linear_power_spectrum(k)])
        linear_power_spectrums_percentage.append([k,(linear_power_spectrum(k)/pk_lcdm[redshift_index-1](k)-1)*100])
        
        nonlinear_power_spectrum_numerical=myie.import_class_pk(relative_path=class_data_path+"z"+str(redshift_index)+"_"+"pk_nl")
        nonlinear_power_spectrum=sp.interpolate.interp1d(nonlinear_power_spectrum_numerical[0], nonlinear_power_spectrum_numerical[1],
                                            fill_value="extrapolate", assume_sorted=True)
        nonlinear_power_spectrums.append([k,nonlinear_power_spectrum(k)])
        nonlinear_power_spectrums_percentage.append([k,(nonlinear_power_spectrum(k)/pk_nl_lcdm[redshift_index-1](k)-1)*100])
    
    """Save data"""
    save_data_to_path_i=save_data_to_path+str(redshift_index-1)+"_"+params.fitting_func.lower()
    myie.save_to_txt_multicolumn(linear_power_spectrums[-len_var_par_values:], path=save_data_to_path_i+"_pk",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
    myie.save_to_txt_multicolumn(linear_power_spectrums_percentage[-len_var_par_values:], path=save_data_to_path_i+"_pk_perc",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
    myie.save_to_txt_multicolumn(nonlinear_power_spectrums[-len_var_par_values:], path=save_data_to_path_i+"_nl_pk",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
    myie.save_to_txt_multicolumn(nonlinear_power_spectrums_percentage[-len_var_par_values:], path=save_data_to_path_i+"_nl_pk_perc",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
    myie.save_to_txt_multicolumn(halo_mass_functions[-len_var_par_values:], path=save_data_to_path_i+"_hmf",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
    
    myie.save_to_txt_sigma8(sigma8_values,path=save_plot_to_path+"sigma8",var_par=params.this_run_specifier_3,var_par_values=var_par_values)
print(f"Output saved at: {save_data_to_path}")

for i in range(len(params.z)):
    legend.append(r"$\Lambda$CDM   z="+str(params.z[i]))
    halo_mass_functions.append([masses,hmf_lcdm[i](masses,params.z[i])])
    nonlinear_power_spectrums.append([k,pk_nl_lcdm[i](k)])

#%%
import plotting_functions as mypl
mypl.plot([nonlinear_power_spectrums,nonlinear_power_spectrums_percentage],
            xlabel=[""                             ,r"$k$  [$h{\rm Mpc}^{-1}$]"],
            ylabel=[r"P$_{\rm{nl}}$($k$)  $[{\rm Mpc}^3 h^{-3}$]",r"Difference $[\%]$" ],
            title=None,
            xscale=["log","log"],
            yscale=["log","linear"],
            ncol=2,
            # dotted=True,
            legend=[legend,None],save=save,save_plot_dir=save_plot_to_path,name=params.fitting_func.lower()+"_nl_pk")

mypl.plot([linear_power_spectrums,linear_power_spectrums_percentage],
            xlabel=["",r"$k$  $[h{\rm Mpc}^{-1}]$"],
            ylabel=[r"P($k$)  $[{\rm Mpc}^3 h^{-3}]$",r"Difference  $[\%]$"],
            title=None,
            xscale=["log","log"],
            yscale=["log","linear"],
            ncol=2,
            # dotted=True,
            legend=[legend,None],save=save,save_plot_dir=save_plot_to_path,name=params.fitting_func.lower()+"_pk")


mypl.plot(f=[halo_mass_functions,mass_functions_percentage],
            xlabel=["",r"Mass  $[M_\odot h^{-1}]$",],
            ylabel=[r"$\frac{dn}{d\ln (M)}$   $[h^3$ ${\rm Mpc}^{-3}]$",r"Difference  $[\%]$"],
            title=None,
            xscale=["log","log"], 
            yscale=["log","linear"],
            ncol=2,
            # dotted=True,
            legend=[legend,None],save=save,save_plot_dir=save_plot_to_path,name=params.fitting_func.lower()+"_mf")
for i,mfp in enumerate(mass_functions_percentage):
    arr=np.round(mfp[0])
    arr=np.array(arr)
    index=np.where(arr//1e11==7)
    print(legend[i],"\t value",round(np.average(mfp[1][index]),1))
                    