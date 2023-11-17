# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:32:41 2023

@author: sebas
"""
"""Class to store the results of the friedmann solver"""
class friedmann_sol:
    def __init__(self):
        self.legend=[]
        
        self.dark_density_evolution_numerical_a=[]
        self.dark_density_parameter_numerical_a=[]
        self.appox_dark_density_evolution_numerical_a=[]
        self.de_eos_numerical_a=[]
        
        self.scale_parameter_numerical_t=[]
        self.hubble_function_numerical_t=[]
        self.rescaled_hubble_functions_numerical_a=[]

        self.matter_density_parameter_numerical_a=[]
        self.rad_density_numerical_a=[]
        
        self.effective_eos_numerical_a=[]
        
        self.universe_age=[]
        self.deceleration_parameter=[]
        
    def append_instance(self, other):
        if isinstance(other, friedmann_sol):
            self.dark_density_evolution_numerical_a.append( other.dark_density_evolution_numerical_a)
            self.dark_density_parameter_numerical_a.append(other.dark_density_parameter_numerical_a)
            self.appox_dark_density_evolution_numerical_a.append(other.appox_dark_density_evolution_numerical_a)
            self.de_eos_numerical_a.append(other.de_eos_numerical_a)
            
            self.scale_parameter_numerical_t.append(other.scale_parameter_numerical_t)
            self.hubble_function_numerical_t.append(other.hubble_function_numerical_t)
            self.rescaled_hubble_functions_numerical_a.append(other.rescaled_hubble_functions_numerical_a)

            self.matter_density_parameter_numerical_a.append(other.matter_density_parameter_numerical_a)
            self.rad_density_numerical_a.append(other.rad_density_numerical_a)
            
            self.effective_eos_numerical_a.append(other.effective_eos_numerical_a)
            self.universe_age.append(other.universe_age)
            self.deceleration_parameter.append(other.deceleration_parameter)
    
    
    
"""Class to store the quintessence results"""
class quintessence_sol:
    def __init__(self):
        self.phi_tilde_numerical_a=[]
        self.approx_phi_tilde_numerical_a=[]
        self.potential_tilde_numerical_a=[]
        self.approx_potential_tilde_numerical_a=[]
        self.potential_tilde_numerical_phi=[]
        self.approx_potential_tilde_numerical_phi=[]
        pass


class perturbation_results:
    def __init__(self):
        self.legend=[]
        
        self.linear_matter_density_contrasts_at_collapse_z=[]
        self.virialization_overdensities_star_z=[]
        self.linear_matter_density_contrast_a=[]
        self.linear_de_density_contrast_a=[]

    def append_instance(self, other):
        if isinstance(other, perturbation_results):
            self.linear_matter_density_contrasts_at_collapse_z.append( other.linear_matter_density_contrasts_at_collapse_z)
            self.virialization_overdensities_star_z.append(other.virialization_overdensities_star_z)
            self.linear_matter_density_contrast_a.append(other.linear_matter_density_contrast_a)
            self.linear_de_density_contrast_a.append(other.linear_matter_de_contrast_a)
