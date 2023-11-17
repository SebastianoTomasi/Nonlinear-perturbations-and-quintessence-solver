# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:22:40 2023

@author: sebas
"""


import scipy as sp
import numpy as np
import sys

import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import matplotlib.pyplot as pl


sys.path.append("../utility_modules")
import lcdm_model as lcdm
import import_export as myie
import cosmological_functions as cosm_func
import plotting_functions as mypl

# %%

save = True
format_=".pdf"#".pdf"

compute_approximations=False

this_run_specifier_0 = "linear_perturbations"
this_run_specifier_1 = "perturbed_de"
this_run_specifier_2 = "de_eos_6"
this_run_specifier_3 = "trans_steepness"

name_specifiers_0 = ["nonlinear_perturbations", "linear_perturbations","quintessence_model"]
name_specifiers_1 = ["plots", "unperturbed_de", "perturbed_de", "LCDM", "EDS"]
name_specifiers_2 = ["de_eos_"+str(i) for i in range(1, 8)]
name_specifiers_3=['w_i','w_f','trans_steepness','trans_z']

#%%

# for this_run_specifier_0 in ["nonlinear_perturbations"]:
#     for this_run_specifier_1 in ["unperturbed_de","perturbed_de"]:
#         for this_run_specifier_2 in ["de_eos_1"]:
#             for this_run_specifier_3 in ['w_i','w_f','trans_steepness','trans_z']:
if this_run_specifier_0=="quintessence_model" and this_run_specifier_3=="trans_z":
    zoomed_in=True
    zoomed_xlim=(0.2,0.26)
    zoomed_ylim=(0.5,3)
    zoomed_position=(0.25, 0.5, #x_pos,y_pos
                      0.4,#lenght 
                      0.4)#height
elif this_run_specifier_0=="linear_perturbations" and this_run_specifier_3=="trans_steepness" and this_run_specifier_2=="de_eos_1":
    zoomed_in=True
    zoomed_xlim=(0.5,0.57)
    zoomed_ylim=(0.42,0.48)
    zoomed_position=(0.55, 0.1, #x_pos,y_pos
                      0.4,#lenght 
                      0.4)#height
  
elif this_run_specifier_0=="linear_perturbations" and this_run_specifier_3=="trans_steepness" and this_run_specifier_2=="de_eos_5":
    zoomed_in=True
    zoomed_xlim=(0.5,0.57)
    zoomed_ylim=(0.46,0.52)
    zoomed_position=(0.55, 0.1, #x_pos,y_pos
                      0.4,#lenght 
                      0.4)#height
elif this_run_specifier_0=="linear_perturbations" and this_run_specifier_3=="trans_z" and this_run_specifier_2=="de_eos_5":
    zoomed_in=True
    zoomed_xlim=(0.5,0.57)
    zoomed_ylim=(0.46,0.52)
    zoomed_position=(0.55, 0.1, #x_pos,y_pos
                      0.4,#lenght 
                      0.4)#height
elif this_run_specifier_0=="linear_perturbations" and this_run_specifier_3=="w_i" and this_run_specifier_2=="de_eos_1":
    zoomed_in=True

    zoomed_xlim=(0.6,0.7)
    zoomed_ylim=(0.51,0.625)
    zoomed_position=(0.55, 0.1, #x_pos,y_pos
                      0.4,#lenght 
                      0.4)#height
else:
    zoomed_in=False


# %%
"""NON-LINEAR PERTURBATIONS"""

if this_run_specifier_0 == "nonlinear_perturbations":

    """Import the LCDM to compare evrithing else aganist."""
    lcdm_linear_matter_density_contrasts_at_collapse_z = myie.import_from_txt_multicolumn(
        "../data/"+this_run_specifier_0+"/LCDM/delta_c")
    lcdm_virialization_overdensities_star = myie.import_from_txt_multicolumn(
        "../data/"+this_run_specifier_0+"/LCDM/zeta_vir_star")

    save_plot_to_path = "../data"+"/"+this_run_specifier_0+"/" + \
        this_run_specifier_1+"/"+this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"
    get_data_from_path = "../data"+"/"+this_run_specifier_0+"/" + \
        this_run_specifier_1+"/"+this_run_specifier_2+"/"+this_run_specifier_3+"/"

    try:
        linear_matter_density_contrasts_at_collapse_z = myie.import_from_txt_multicolumn(
            get_data_from_path+"delta_c")
        virialization_overdensities_star = myie.import_from_txt_multicolumn(
            get_data_from_path+"zeta_vir_star")
        effective_eos_a = myie.import_from_txt_multicolumn(
            get_data_from_path + "effective_eos")
        omega_matter_a = myie.import_from_txt_multicolumn(
            get_data_from_path+"omega_matter")
        growth_factr = myie.import_from_txt_multicolumn(
            get_data_from_path+"growth_factor")
    except FileNotFoundError:
        print(f"Error: file not found. Check if the data is available at {get_data_from_path}")
    
    """Append the lcdm model curves"""
    linear_matter_density_contrasts_at_collapse_z.append(
        lcdm_linear_matter_density_contrasts_at_collapse_z[0])
    virialization_overdensities_star.append(
        lcdm_virialization_overdensities_star[0])


    """Set up the legend"""
    params_names = {"w_i": "w_i",
                    "w_f": "w_f",
                    "trans_steepness": "\Gamma",
                    "trans_z": "z_t",
                    "de_eos_a": this_run_specifier_2}

    """Import the values of the varied parameter and build the legend"""
    var_par_values = myie.import_varied_parameter_values(
        get_data_from_path+"delta_c", this_run_specifier_3)
    legend=[]
    if this_run_specifier_2=="de_eos_6":
        legend=["Model 6"]
    else:
        for i in var_par_values:
            legend.append("$"+params_names[this_run_specifier_3]+"$="+str(round(i, 3)))
    legend.append("$\Lambda CDM$")

    """Do the actual plotting"""
    if zoomed_in:
        mypl.plot(linear_matter_density_contrasts_at_collapse_z, legend=legend,
                    # title="Linearly extrapolated matter density contrast at collapse.",
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"$\delta_{\rm{c}}$",
                    save=save,
                    name="zoomed_delta_c",
                    save_plot_dir=save_plot_to_path, format_=format_,
                    zoomed=True, 
                    zoomed_xlim=zoomed_xlim,zoomed_xticks=zoomed_xlim,
                    zoomed_ylim=zoomed_ylim,zoomed_yticks=zoomed_ylim, 
                    zoomed_position=zoomed_position)

        # mypl.plot(virialization_overdensities_star, legend=legend,
        #             # title="Matter overdensity at virialization.",
        #             title="",
        #             xlabel="$z_{\rm{c}}$", ylabel="$\zeta_{\rm{vir}}^*$",
        #             save=save,
        #             name="zoomed_zeta_vir_star",
        #             save_plot_dir=save_plot_to_path, format_=format_,
        #             zoomed=True, 
        #             zoomed_xlim=zoomed_xlim,zoomed_xticks=zoomed_xlim,
        #             zoomed_ylim=zoomed_ylim,zoomed_yticks=zoomed_ylim, 
        #             zoomed_position=zoomed_position)
    else:
        delta_c_percentage_difference_wrt_lcdm=[]#
        for delta in linear_matter_density_contrasts_at_collapse_z:
            delta_c_percentage_difference_wrt_lcdm.append([delta[0],(np.array(delta[1])/np.array(lcdm_linear_matter_density_contrasts_at_collapse_z[0][1])-1)*100])
            
        mypl.plot(delta_c_percentage_difference_wrt_lcdm, legend=legend,
                    # title="Linearly extrapolated matter density contrast at collapse.",
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"Difference  $[\%]$",
                    save=save,
                    name="delta_c_percentage_difference_wrt_lcdm",
                    save_plot_dir=save_plot_to_path, format_=format_)
        
        
        mypl.plot(linear_matter_density_contrasts_at_collapse_z, legend=legend,
                    # title="Linearly extrapolated matter density contrast at collapse.",
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"$\delta_{\rm{c}}$",
                    save=save,
                    name="delta_c",
                    save_plot_dir=save_plot_to_path, format_=format_)
    
        mypl.plot(virialization_overdensities_star, legend=legend,
                    # title="Matter overdensity at virialization.",
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"$\zeta_{\rm{vir}}^*$",
                    save=save,
                    name="zeta_vir_star",
                    save_plot_dir=save_plot_to_path, format_=format_)
    
        mypl.plot(effective_eos_a, legend=legend,
                    # title="Effective eos",
                    title="",
                    func_to_compare=lcdm.effective_eos_a,
                    xlabel=r"$a$", ylabel=r"$w_{\rm{eff}}$",
                    save=save,
                    name="effective_eos",
                    save_plot_dir=save_plot_to_path, format_=format_)
    
    
        
        zeta_percentage_difference_wrt_lcdm=[]#model=lcdm(1+%)
        for i in range(len(virialization_overdensities_star)-1):
            zeta_percentage_difference_wrt_lcdm.append([virialization_overdensities_star[i][0],
                                       (np.array(virialization_overdensities_star[i][1])/np.array(virialization_overdensities_star[-1][1])-1)*100])
        
        maxima_points_x,maxima_points_y=[],[]#Compute the points of maximum
        for element in zeta_percentage_difference_wrt_lcdm:
            max_value=float(max(element[1]))
            max_index=np.where(element[1]==max_value)
            maxima_points_x.append(round(float(element[0][max_index]),2))
            maxima_points_y.append(round(max_value,2))
        print(maxima_points_x,maxima_points_y)
            
        mypl.plot(zeta_percentage_difference_wrt_lcdm, legend=legend,
                    # title="Matter density parameter.",
                    x_ticks=maxima_points_x,
                    y_ticks=maxima_points_y,
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"Difference  $[\%]$",
                    save=save,
                    name="zeta_percentage_difference_wrt_lcdm_wmp",
                    save_plot_dir=save_plot_to_path, format_=format_)
            
        mypl.plot(zeta_percentage_difference_wrt_lcdm, legend=legend,
                    # title="Matter density parameter.",
                    title="",
                    xlabel=r"$z_{\rm{c}}$", ylabel=r"Difference  $[\%]$",
                    save=save,
                    name="zeta_percentage_difference_wrt_lcdm",
                    save_plot_dir=save_plot_to_path, format_=format_)
        
    if save:
        header=myie.import_header(get_data_from_path+"/delta_c")
        myie.save_header(header, save_plot_to_path+"simulation_parameters")

#%%
"""LINEAR PERTURBATIONS"""

if this_run_specifier_0 == "linear_perturbations":

    save_plot_to_path = "../data/nonlinear_perturbations/" + \
        this_run_specifier_1+"/"+this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"
    get_data_from_path = "../data/nonlinear_perturbations/" + \
        this_run_specifier_1+"/"+this_run_specifier_2+"/"+this_run_specifier_3+"/"

    try:
        dark_energy_eos = myie.import_from_txt_multicolumn(get_data_from_path+"dark_energy_eos")
        delta_m = myie.import_from_txt_multicolumn(get_data_from_path+"growth_factor")
        delta_de = myie.import_from_txt_multicolumn(get_data_from_path+"dark_growth_factor")
        effective_eos_a = myie.import_from_txt_multicolumn(get_data_from_path + "effective_eos")
        # growth_evolution = myie.import_from_txt_multicolumn(get_data_from_path+"growth_evolution")
        omega_matter = myie.import_from_txt_multicolumn(get_data_from_path+"omega_matter")
    except FileNotFoundError:
        print(f"Error: file not found. Check if the data is available at {get_data_from_path}")

    """PLOTTING """
    
    """Set up the legend"""
    params_names = {"w_i": "w_i",
                    "w_f": "w_f",
                    "trans_steepness": "\Gamma",
                    "trans_z": "z_t",
                    "de_eos_a": this_run_specifier_2}

    """Import the values of the varied parameter and build the legend"""
    var_par_values = myie.import_varied_parameter_values(
        get_data_from_path+"growth_factor", this_run_specifier_3)
    legend = []
    for i in var_par_values:
        legend.append(
            "$"+params_names[this_run_specifier_3]+"$="+str(round(i, 3)))
    if this_run_specifier_2=="de_eos_6":
        legend=["Model_6"]
    legend.append("$\Lambda CDM$")


    """Use the lcdm_model to do comparisons"""
    LCDM = lcdm.LCDM()
    linear_perturbations_lcdm_numerical_a = LCDM.compute_perturbations()
    lcdm_scale_parameter = linear_perturbations_lcdm_numerical_a[0]
    linear_perturbations_lcdm_a = sp.interpolate.interp1d(lcdm_scale_parameter,
                                                          linear_perturbations_lcdm_numerical_a[1],
                                                          fill_value="extrapolate", assume_sorted=False)

    """EQUATIONS OF STATE"""
    mypl.plot(dark_energy_eos, r"$a$", r"$w$",
                title="",
                legend=legend,
                # func_to_compare=LCDM.cosmological_constant_eos_a,
                save=save,
                save_plot_dir=save_plot_to_path, format_=format_,
                # xscale="log",
                name="de_eos",
                yscale="linear")
    


    """EFFECTIVE EOS"""
    mypl.plot(effective_eos_a, r"$a$", r"$w_{\rm{eff}}$",
                title="",
                legend=legend,
                func_to_compare=LCDM.effective_eos_a,
                save=save,
                save_plot_dir=save_plot_to_path, format_=format_,
                # xlim=(0,1/(z_t+1)),
                # xscale="log",
                name="effective_eos")
    

    """MATTER DENSITY PARAMETERS"""
    mypl.plot(omega_matter, r"$a$", r"$\Omega_{\rm{m}}$",
                title="",
                legend=legend,
                func_to_compare=LCDM.omega_matter_a,
                save=save,
                save_plot_dir=save_plot_to_path, format_=format_,
                name="omega_matter_detail",
                xscale="log",
                yscale="linear",
                zoomed=True, 
                zoomed_xlim=(0.05,0.6),zoomed_xticks=(0.05,0.6),
                zoomed_ylim=(0.65,1.1),zoomed_yticks=(0.65,1.1), 
                zoomed_position=(0.25, 0.1, #x_pos,y_pos
                                  0.55,#lenght 
                                  0.55))
    

    if this_run_specifier_1=="perturbed_de":
        mypl.plot(delta_de, r"$a$", r"$\delta_{\rm{de}}$",
                    title="",
                    legend=legend,
                    func_to_compare=None,
                    save=save,
                    save_plot_dir=save_plot_to_path, format_=format_,
                    dotted=False,
                    name="dark_contrast_matter_a",
                    )
        

    """Matter perturbations on the total energy density"""
    matter_density_perturbations_hat_a = []
    for i in range(len(delta_m)):
        a = delta_m[i][0]
        omega_m = sp.interpolate.interp1d(omega_matter[i][0],
                                          omega_matter[i][1],
                                          fill_value="extrapolate", assume_sorted=True)
        
        # omega_de = sp.interpolate.interp1d(omega_dark[i][0],
        #                                     omega_dark[i][1],
        #                                     fill_value="extrapolate", assume_sorted=True)
        matter_density_perturbations_hat_a.append([a,
                                                    omega_m(a)*delta_m[i][1]])
    mypl.plot(matter_density_perturbations_hat_a, r"$a$", r"$\Omega_{\rm{m}}\delta_{\rm{m}}$",
                title="",
                legend=legend,
                func_to_compare=None,
                save=save,
                save_plot_dir=save_plot_to_path, format_=format_,
                dotted=False,
                name="total_density_contrast",
                )
    

    """MATTER DENSITY CONTRAST"""
    # delta_m.append(linear_perturbations_eds_numerical_a)
    mypl.plot(delta_m, r"$a$", r"$\delta_{\rm{m}}$",
                title="",
                legend=legend,
                func_to_compare=linear_perturbations_lcdm_a,
                save=save,
                save_plot_dir=save_plot_to_path, format_=format_,
                dotted=False,
                name="density_contrast_matter_a",
                # xlim=(0.9,1),ylim=(0.7,0.85)
                )
    

    """GROWTH EVOLUTION as a function of z"""
    linear_perturbations_lcdm_z=sp.interpolate.interp1d(cosm_func.z_a(lcdm_scale_parameter),
                                    linear_perturbations_lcdm_numerical_a[1]/(linear_perturbations_lcdm_numerical_a[1][-1]*lcdm_scale_parameter),
                                        fill_value="extrapolate", assume_sorted=False)

    # mypl.plot(growth_evolution,"$z$","$D_+$",
    #             title="",
    #         legend=legend,
    #         func_to_compare=linear_perturbations_lcdm_z,
    #         xlim=(1e-1,1e3),
    #         save=save,
    #         # dotted=True,
    #         save_plot_dir=save_plot_to_path, format_=format_,
    #         name="growth_evolution_matter_z",
    #         xscale="log"
    #         )
    if zoomed_in:
        mypl.plot(delta_m, r"$a$", r"$\delta_{\rm{m}}$",
                    title="",
                    legend=legend,
                    func_to_compare=linear_perturbations_lcdm_a,
                    save=save,
                    save_plot_dir=save_plot_to_path, format_=format_,
                    dotted=False,
                    name="detail_density_contrast_matter_a",
                    # xlim=(0.9,1),ylim=(0.7,0.85),
                    zoomed=True, 
                    zoomed_xlim=zoomed_xlim,zoomed_xticks=zoomed_xlim,
                    zoomed_ylim=zoomed_ylim,zoomed_yticks=zoomed_ylim, 
                    zoomed_position=zoomed_position
                    )


    if save:
        header=myie.import_header(get_data_from_path+"/growth_factor")
        myie.save_header(header, save_plot_to_path+"simulation_parameters")
#%%
"""QUINTESSENCE MODEL"""      

if this_run_specifier_0=="quintessence_model":
    save_plot_to_path = "../data"+"/"+this_run_specifier_0+"/"+ \
        this_run_specifier_2+"/plots/"+this_run_specifier_3+"/"
    get_data_from_path = "../data"+"/"+this_run_specifier_0+"/" + \
        this_run_specifier_2+"/"+this_run_specifier_3+"/"

    try:
        phi_tilde_numerical_a = myie.import_from_txt_multicolumn(get_data_from_path+"phi_tilde_a")
        approx_phi_tilde_numerical_a = myie.import_from_txt_multicolumn(get_data_from_path+"approx_phi_tilde_a")
        potential_tilde_numerical_a = myie.import_from_txt_multicolumn(get_data_from_path+"potential_tilde_a")
        approx_potential_tilde_numerical_a = myie.import_from_txt_multicolumn(get_data_from_path+"approx_potential_tilde_a")
        dark_energy_eos = myie.import_from_txt_multicolumn(get_data_from_path + "eos")
        approx_dark_energy_eos = myie.import_from_txt_multicolumn(get_data_from_path + "approx_eos")
        potential_tilde_numerical_phi = myie.import_from_txt_multicolumn(get_data_from_path+"potential_tilde_phi")
        approx_potential_tilde_numerical_phi = myie.import_from_txt_multicolumn(get_data_from_path+"approx_potential_tilde_phi")
    except FileNotFoundError:
        print(f"Error: file not found. Check if the data is available at {get_data_from_path}")

    """PLOTTING """
    
    """Set up the legend"""
    params_names = {"w_i": "w_i",
                    "w_f": "w_f",
                    "trans_steepness": "\Gamma",
                    "trans_z": "z_t",
                    "de_eos_a": this_run_specifier_2}

    """Import the values of the varied parameter and build the legend"""
    var_par_values = myie.import_varied_parameter_values(
        get_data_from_path+"eos", this_run_specifier_3)
    legend = []
    for i in var_par_values:
        legend.append(
            "$"+params_names[this_run_specifier_3]+"$="+str(round(i, 3)))
    
    
    """SCALAR FIELD TILDE of a """
    mypl.plot(phi_tilde_numerical_a,
                xlabel=r"$a$",
                ylabel=r"$\tilde{\phi}(a)$",
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
                xlim=None,ylim=(0.1,1e9), dotted=False)
    
    
    
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
                xlim=None,ylim=(0.1,1e6), dotted=False)
    
    
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
    
    
    
    if compute_approximations:
        if this_run_specifier_3=="trans_steepness":
            number=1
        else:
            number=len(approx_phi_tilde_numerical_a)
            
        approx_legend=legend[:]
        for i in range(number):
            if this_run_specifier_3=="trans_steepness":
                approx_legend.append("Approx.")
            else:
                approx_legend.append("Approx. "+legend[i])
            
        
        aux0=phi_tilde_numerical_a[:]
        for i in range(1):
            aux0.append(approx_phi_tilde_numerical_a[i])
            
        """approximated SCALAR FIELD TILDE of a """
        mypl.plot(aux0,
                    xlabel=r"$a$",
                    ylabel=r"$\tilde{\phi} (a)$",
                    title="",
                    legend=approx_legend,
                    save_plot_dir=save_plot_to_path,ncol=2,
                    # location="upper left",
                    save=save,name="approx_scalar_field_tilde_a",
                    xscale="linear",yscale="linear",
                    xlim=None,ylim=None, dotted=False)
        
         
        aux1=potential_tilde_numerical_phi[:]
        for i in range(number) :
            aux1.append(approx_potential_tilde_numerical_phi[i])
              
        """approximated POTENTIAL TILDE of phi"""
        mypl.plot(aux1,
                    xlabel=r"$\tilde{\phi}$",
                    ylabel=r"$\tilde{V}(\tilde{\phi})$",
                    title="",
                    legend=approx_legend,
                    save_plot_dir=save_plot_to_path,ncol=2,
                    save=save,name="approx_potential_tilde_phi",xscale="linear",yscale="log",
                    xlim=None,ylim=(0.1,1e9), dotted=False)
        
        
        """approximated EOS"""
        aux2=dark_energy_eos[:]
        for i in range(1) :
            aux2.append(approx_dark_energy_eos[i])
              
        """approximated EOS"""
        mypl.plot(aux2,
                    xlabel=r"$a$",
                    ylabel=r"$w$",
                    title="",
                    legend=approx_legend,ncol=2,
                    # location="center",
                    save_plot_dir=save_plot_to_path,
                    save=save,name="approx_dark_energy_eos",xscale="linear",yscale="linear",
                    xlim=None,ylim=None, dotted=False)
        
        
        
        if zoomed_in:
            mypl.plot(aux1, 
                        xlabel=r"$\tilde{\phi}$",
                        ylabel=r"$\tilde{V}(\tilde{\phi})$",
                        title="",
                        legend=approx_legend,
                        func_to_compare=None,
                        save=save,
                        save_plot_dir=save_plot_to_path, format_=format_,
                        dotted=False,
                        name="detail_approx_potential_tilde_numerical_phi",xscale="linear",yscale="log",
                        # xlim=(0.9,1),
                        ylim=(0.1,1e9),ncol=1,
                        zoomed=True, 
                        zoomed_xlim=zoomed_xlim, 
                        zoomed_ylim=zoomed_ylim, 
                        zoomed_position=zoomed_position,
                        zoomed_dotted=True
                        )
            pl.show()
        if save:
            header=myie.import_header(get_data_from_path+"/potential_tilde_phi")
            myie.save_header(header, save_plot_to_path+"simulation_parameters")
 

