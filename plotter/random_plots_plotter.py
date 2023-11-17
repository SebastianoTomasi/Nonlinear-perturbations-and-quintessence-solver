# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:49:58 2023

@author: sebas
"""

import numpy as np
from numpy import sin
from numpy import cos
from numpy import log
from numpy import exp
from numpy import e
from numpy import pi
from numpy import sqrt
from numpy import sinh
from numpy import cosh
from numpy import arcsinh
from numpy import arccosh

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
import dark_energy_eos as eos

sys.path.append("../utility_modules")
import numerical_methods as mynm
import plotting_functions as mypl

sys.path.append("../friedmann_solver")
sys.path.append("../perturbation_solver")


#%%

save=True


def zeta(alfa):
    zeta=9/2*(alfa-sin(alfa))**2/(1-cos(alfa))**3
    return zeta

# alfa=np.linspace(0,2*pi-pi/3,150)
# mypl.plot([alfa,zeta(alfa)],title="Matter overdensity in EdS",
#           xlabel=r"$\alpha$",ylabel=r"$\zeta$",
#           yscale="log",
#           legend=[None],
#           x_ticks=[0,np.pi,3*np.pi/2],
#           x_ticklabels=["0", r"$\pi$", r"$\frac{3 \pi }{2 }$"],
#           y_ticks=[1,5.552,146.85], y_ticklabels=[1,5.552,146.85],
#           name="zeta_vir_eds",save=save)

#%%
def E_wcdm(a,w):
    return np.sqrt(params.omega_matter_now/a**3+params.omega_dark_now*a**(-3*(1+w)))

def E_lcdm(a):
    return E_wcdm(a,-1)

def E_eds(a):
    return E_wcdm(a,0)


a=np.linspace(1e-10, 1,500)
w=np.linspace(-0.5, -0.9,3)

def fun_to_plot(x,*kwargs):
    fun_to_plot= (E_wcdm(x,kwargs[0])/E_lcdm(x)-1)*100
    return fun_to_plot

f=[[cosm_func.z_a(a),fun_to_plot(a,val)] for val in w ]
legend=["$w$="+str(round(val,1)) for val in w]
mypl.plot(f, xlabel=r"$z$",ylabel= r"Difference  $[\%]$",
          title="",
          legend=legend,
          func_to_compare=None,
          save=save, format_=".pdf",
          dotted=False,
            xlim=(-1,params.max_redshift),
          name="hubble_ratio_de_lcdm_z")



f=[[a,fun_to_plot(a,val)] for val in w ]
legend=[r"$w$="+str(round(val,1)) for val in w]
mypl.plot(f, r"$a$", r"Difference $[\%]$",
          title="",
          legend=legend,
          func_to_compare=None,
          save=save, format_=".pdf",
          dotted=False,
          name="hubble_ratio_de_lcdm")



def fun_to_plot(x,*kwargs):
    fun_to_plot= (E_wcdm(x,kwargs[0])/(np.sqrt(params.omega_matter_now)*E_eds(x))-1)*100
    return fun_to_plot
f=[[cosm_func.z_a(a),fun_to_plot(a,val)] for val in w ]
legend=[r"$w$="+str(round(val,1)) for val in w]
mypl.plot(f, r"$z$", r"Difference $[\%]$",
          title="",
           # xscale="log",
          legend=legend,
          save=save,
          func_to_compare=None,
          format_=".pdf",
          dotted=False,
            xlim=(params.min_redshift,params.max_redshift),
          name="hubble_ratio_de_eds")

#%%
a=np.linspace(0,1,200)
z=cosm_func.z_a(a)

all_de_eos=[]


w_i=0.5
w_f=-0.7
z_t=0.5
gamma=10
legend=[]

for i in range(1,7):
    print(w_i,i)
    if i==2:
        gamma=5
    if i==3:
        gamma=0.1
    if i==4:
        gamma=0.04
    if i==5:
        gamma=-1
    args={"w_i":w_i,"w_f":w_f,"z_t":z_t,"gamma":gamma}
    exec("equazione=eos.de_eos_"+str(i))
    legend.append(str(i))
    all_de_eos.append([a,equazione(z,**args)])
    w_i+=-0.3
    # w_f+=-0.2

mypl.plot(all_de_eos,legend=legend,
          ylabel=r"$w$",xlabel=r"$a$",title="",
          name="all_de_eos",save=save)

#%%
# w=np.linspace(-0.8,-1,10)
def dec_wcdm(a,w):
    return 1/(2*a**3*E_wcdm(a, w))*((1+3*w)*params.omega_dark_now*a**(-3*w)+params.omega_matter_now)

def dec_lcdm(a):
    return dec_wcdm(a,-1)
z=np.linspace(0.1, 50,int(1e3))

f=[[z,(dec_wcdm(cosm_func.a_z(z),val)/dec_lcdm(cosm_func.a_z(z))-1)*100] for val in w ]

legend=["$w$="+str(round(val,1)) for val in w]
legend.append(r"$\Lambda$CDM")
mypl.plot(f, xlabel=r"$z$",ylabel= r"Acceleration",
          title="",
           # legend=legend,
            xscale="log",
            # yscale="log",
            func_to_compare=lambda z: z/z*0,
          save=save, format_=".pdf",
          # dotted=True,
            # xlim=(0,50),
            # ylim=(-10,10),
          name="acc_percentange_wcdm_lcdm")

f=[]
f=[[z,dec_wcdm(cosm_func.a_z(z),val)] for val in w ]
f.append([z,dec_lcdm(cosm_func.a_z(z))])

mypl.plot(f, xlabel=r"$z$",ylabel= r"Acceleration",
          title="",
            legend=legend,
            xscale="log",
            # yscale="log",
            # func_to_compare=lambda z: z/z*0,
          save=save, format_=".pdf",
          # dotted=True,
            # xlim=(0,50),
            ylim=(-1,10),
          name="acceleration_wcdm")






