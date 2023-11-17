# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:51:30 2023

@author: Sebastiano Tomasi
"""
import matplotlib.pyplot as pl
import numpy as np



import os
# Get the absolute path of the script Python file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory
os.chdir(script_dir)

import sys
print(sys.executable)
sys.path.append("./nonlinear_perturbations_solver")
import dark_energy_spherical_collapse_analysis as desca
import pseudo_newtonian_perturbations as psp

sys.path.append("./quintessence_solver")
# import scalar_field_potential_analysis as sfpa

sys.path.append("./utility_modules")
import plotting_functions as mypl
import import_export as myie

sys.path.append("./data_modules")
import cosmological_functions as cosm_func
import simulation_parameters as params

sys.path.append("./friedmann_solver")
import friedmann_solver as solve_background

only_linear=True
# bakground_results=solve_background.solve(True,True)
perturbation_results,background_results=desca.solve(only_linear)
# quintessence_results=sfpa.solve()

save=False
if not only_linear:
    """Linear matter density contrast at collapse"""
    mypl.plot(perturbation_results.linear_matter_density_contrasts_at_collapse_z,"$z$","$\delta_c(z)$",
            "Linear density contrast at collapse",
            legend=perturbation_results.legend,
            # func_to_compare=cosm_func.delta_c_eds,
            save=save,dotted=False,
            name="linear_matter_density_contrast_collapse_z",
            )
    pl.show()
                


    mypl.plot(perturbation_results.virialization_overdensities_star_z,"$z$","$\zeta_{vir}^*(z)$",
            "Overdensity at virialization",
            legend=perturbation_results.legend,
            save=save,dotted=False,
            name="overdensity_at_virialization_star",
            )
    pl.show()

mypl.plot(perturbation_results.linear_de_density_contrast_a,r"$a$",r"$\delta_{de}(a)$",
        title="",
        legend=perturbation_results.legend,
        # func_to_compare=cosm_func.delta_c_eds,
        save=save,dotted=False,
        name="linear_de_density_contrast_collapse_a",
        )
pl.show()
mypl.plot(perturbation_results.linear_matter_density_contrast_a,"$a$","$\delta_m(a)$",
        "",
        legend=perturbation_results.legend,
        # func_to_compare=cosm_func.delta_c_eds,
        save=save,dotted=False,
        name="linear_matter_density_contrast_collapse_a",
        )
pl.show()







