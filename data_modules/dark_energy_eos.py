# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:42:50 2022

@author: sebas
"""
import numpy as np
from numpy import log
from numpy import log10
from numpy import exp
from numpy import tanh

def wcdm(z, **kwargs):
    w = kwargs["w"]
    if np.isscalar(z):
        return w
    else:
        return np.full_like(z, w)

def cpl(z, **kwargs):
    w_i, w_f = kwargs["w_i"], kwargs["w_f"]
    return w_f + (z/(1+z))*w_i

def de_eos_1(z, **kwargs):
    w_i, w_f, z_t, gamma = kwargs["w_i"], kwargs["w_f"], kwargs["z_t"], kwargs["gamma"] 
    return 0.5*(w_i+w_f) - 0.5*(w_i-w_f)*tanh(gamma*log((1+z_t)/(1+z)))

def de_eos_2(z, **kwargs):
    w_i, w_f, z_t, gamma = kwargs["w_i"], kwargs["w_f"], kwargs["z_t"], kwargs["gamma"]
    q = gamma
    return w_f + (w_i-w_f)*(z/z_t)**q/(1+(z/z_t)**q)

def de_eos_3(z, **kwargs):
    w_i, w_f, z_t, delta = kwargs["w_i"], kwargs["w_f"], kwargs["z_t"], kwargs["gamma"]
    return w_i + (w_f-w_i)/(1+exp((z-z_t)/delta))


def de_eos_4(z, **kwargs):
    w_i, w_f, z_t, gamma = kwargs["w_i"], kwargs["w_f"], kwargs["z_t"], kwargs["gamma"]
    a = 1 / (z + 1)
    a_t = 1 / (z_t + 1)
    return w_f + (w_i - w_f) * (1 + exp(a_t / gamma)) / (1 + exp((-a + a_t) / gamma)) * (1 - exp((-a + 1) / gamma)) / (1 - exp(1 / gamma))

def de_eos_5(z, **kwargs):
    z_t, gamma = kwargs["z_t"], kwargs["gamma"] 
    return -gamma / (3 * log(10)) * (1 + tanh(gamma * log10((1 + z) / (1 + z_t)))) - 1

def de_eos_6(z, **kwargs):
    return -1 / (3 * log(10)) * (1 + tanh(log10(1 + z))) - 1
