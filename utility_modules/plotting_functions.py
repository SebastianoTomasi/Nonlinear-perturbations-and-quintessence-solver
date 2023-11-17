# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:15:11 2022

@author: sebas
"""


import matplotlib.pyplot as pl
from matplotlib.lines import Line2D


from datetime import datetime
import numpy as np

import sys
sys.path.append("../data_modules")
import simulation_parameters as params

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True



default_save_plot_dir="../data/defoult_plot_folder"





def dots(dotted):
    if dotted:
        return "o"
    else:
        return ""


def print_dict(D):
    for key,value in D.items():
        print(key+"=",value)
        


def plot(f,
         xlabel=["$x$"], ylabel=["$y$"],
         title="Function plot",
         legend=[("First", "Second")],
         func_to_compare=None,
         save=False, save_plot_dir=default_save_plot_dir, name=None, format_=".pdf",
         xscale=["linear","linear"], yscale=["linear","linear"],
         xlim=None, ylim=None,
         dotted=False,
         x_ticks=[], x_ticklabels=[], y_ticks=[], y_ticklabels=[],
         zoomed=False, zoomed_xlim=None, zoomed_ylim=None, zoomed_position=(0.2, 0.55, 0.3, 0.3),
         zoomed_xticks=[], zoomed_yticks=[],
         skip_first=False,ncol=1,location="best",zoomed_dotted=False):
    
    """Everithing must be vectorized to allow for stacked plots (one plot on top
    of the other.)"""
    if isinstance(xlabel, str):
        xlabel=[xlabel]
    if isinstance(ylabel, str):
        ylabel=[ylabel]
    if isinstance(legend[0],str):
         legend=[legend]
    if isinstance(xscale, str):
         xscale=[xscale]
    if isinstance(yscale, str):
         yscale=[yscale]
    if xlim!=None:
        if isinstance(xlim, tuple):
             xlim=[xlim]
    if ylim!=None:
        if isinstance(ylim, tuple):
             ylim=[ylim]
             
    """Colors and linestiles used. The number of colors and linestiles should be
    relative prime numbers to maximize the number of possilbe combinations."""
    colors = "bgcmyk"
    linestyle_str = ['-', '--', '-.', ':']
    line_width=3
    xy_labels_fontsize=28
    legend_fontsize=20
    labelsize=20
    title_fontsize=16
    
    """Check if the input is a single plot [x,y], a multi plot[[x_1,y_1],[x_2,y_2]..]
    or a stacked plot.[[multi plot 1],[multi plot 2]]"""
    is_singleplot = False
    stacked_plots=True
    try:
        f[0][0][0]
    except (IndexError, TypeError):#Index for numpy floats and Type for standard float
        is_singleplot = True
    try:
        f[0][0][0][0]
    except(IndexError, TypeError):
        stacked_plots=False
        
    """Distinghish if the user wants two plots stacked or one."""
    if stacked_plots:
        fig, ax = pl.subplots(nrows=2, ncols=1, figsize=(12, 14), dpi=100,sharex=True)
    else:
        fig, ax = pl.subplots(figsize=(10, 6), dpi=100)
        ax=[ax]#Vectorize in order to make the same code work for stacked_plots=True

    # Create a list to hold the custom handles
    handles = []

    for i in range(len(ax)):
        ax[i].set_xscale(xscale[i])
        ax[i].set_yscale(yscale[i])
        if is_singleplot:
            ax[0].plot(f[0], f[1], colors[i % 6] + dots(dotted), linestyle=linestyle_str[i % 4],linewidth=line_width)
            if zoomed:
                axins = ax[i].inset_axes(zoomed_position)
                axins.plot(f[0], f[1],"b",linewidth=line_width)
                axins.set_xlim(zoomed_xlim)
                axins.set_ylim(zoomed_ylim)
                ax[i].indicate_inset_zoom(axins)
        elif stacked_plots:
            for j in range(len(f[i])):
                ax[i].plot(f[i][j][0], f[i][j][1], colors[j % 6] + dots(dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
                # Create a custom handle for this line
                line = Line2D([0], [0], color=colors[j % 6], linestyle=linestyle_str[j % 4], linewidth=line_width)
                handles.append(line)
        else:
            axins = ax[i].inset_axes(zoomed_position)   if zoomed else None
            for j in range(len(f)):
                ax[i].plot(f[j][0], f[j][1], colors[j % 6] + dots(dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
                if zoomed:
                    axins.plot(f[j][0], f[j][1], colors[j % 6] + dots(zoomed_dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
                    axins.set_xlim(zoomed_xlim)
                    axins.set_ylim(zoomed_ylim)
                    ax[i].indicate_inset_zoom(axins)

                # Create a custom handle for this line
                line = Line2D([0], [0], color=colors[j % 6], linestyle=linestyle_str[j % 4], linewidth=line_width)
                handles.append(line)
        ax[i].set_title(title, fontsize=title_fontsize)

        ax[i].set_xlabel(xlabel[i], fontsize=xy_labels_fontsize)
        ax[i].set_ylabel(ylabel[i], fontsize=xy_labels_fontsize)
        
        ax[i].tick_params(axis='both', which='both', labelsize=labelsize)
        
        if func_to_compare is not None:
            x = np.linspace(*ax[i].get_xlim())
            function = np.vectorize(func_to_compare)
            ax[i].plot(x, function(x), "r")
            line = Line2D([0], [0], color="r", linewidth=line_width)
            handles.append(line)
        
        if xlim is not None:
            ax[i].set_xlim(xlim[i])
        if ylim is not None:
            ax[i].set_ylim(ylim[i])
        
        if len(x_ticks) != 0:
            ax[i].set_xticks(x_ticks)
            if len(x_ticklabels)!=0:
                ax[i].set_xticklabels(x_ticklabels)
            else:
                for xtick in x_ticks:
                    x_ticklabels.append(str(xtick))
                ax[i].set_xticklabels(x_ticklabels)
                    
        if len(y_ticks) != 0:
            ax[i].set_yticks(y_ticks)
            if len(y_ticklabels)!=0:
                ax[i].set_yticklabels(y_ticklabels)
            else:
                for ytick in y_ticks:
                    y_ticklabels.append(str(ytick))
                ax[i].set_yticklabels(y_ticklabels)

        # Add the legend with the custom handles
        if legend[i]!=None:
            ax[i].legend(handles, legend[i], shadow=True, loc=location, handlelength=1.5, fontsize=legend_fontsize,ncol=ncol)
    pl.tight_layout()
    
    if save==True:
        dpi=80
        now = datetime.now()
        current_time = now.strftime("%Y_%m")
        if name!=None:
            pl.savefig(save_plot_dir+"/"+name+"_"+current_time+format_,dpi=dpi,bbox_inches='tight')
        else:
            pl.savefig(save_plot_dir+"/"+"_"+current_time+format_,dpi=dpi,bbox_inches='tight')

    pl.show()
    return None  

# def g(x):
#     return x**2
# def h(x):
#     return x
# x=np.linspace(0,5,10)
# m_plot([x,g(x)],dotted=True)
# pl.show()
# m_plot([[x,g(x)],[x,h(x)]],dotted=True)
# pl.show()

# def plot_older(f,
#          xlabel=["$x$"], ylabel=["$y$"],
#          title="Function plot",
#          legend=[("First", "Second")],
#          func_to_compare=None,
#          save=False, save_plot_dir=default_save_plot_dir, name=None, format_=".pdf",
#          xscale=["linear","linear"], yscale=["linear","linear"],
#          xlim=None, ylim=None,
#          dotted=False,
#          x_ticks=[], x_ticklabels=[], y_ticks=[], y_ticklabels=[],
#          zoomed=False, zoomed_xlim=None, zoomed_ylim=None, zoomed_position=(0.2, 0.55, 0.3, 0.3),
#          zoomed_xticks=[], zoomed_yticks=[],
#          skip_first=False,ncol=1,location="best",zoomed_dotted=False):
    
#     if isinstance(xlabel, str):
#         xlabel=[xlabel]
#     if isinstance(ylabel, str):
#         ylabel=[ylabel]
#     if isinstance(legend[0],str):
#          legend=[legend]
#     if isinstance(xscale, str):
#          xscale=[xscale]
#     if isinstance(yscale, str):
#          yscale=[yscale]
#     if xlim!=None:
#         if isinstance(xlim, tuple):
#              xlim=[xlim]
#     if ylim!=None:
#         if isinstance(ylim, tuple):
#              ylim=[ylim]
    
    

#     colors = "bgcmyk"
#     linestyle_str = ['-', '--', '-.', ':']
#     line_width=3
#     xy_labels_fontsize=20
#     legend_fontsize=18
#     labelsize=20
#     title_fontsize=16

#     is_singleplot = False
#     stacked_plots=True
#     try:
#         f[0][0][0]
#     except:
#         is_singleplot = True

#     try:
#         f[0][0][0][0]
#     except:
#         stacked_plots=False
    
    
#     if stacked_plots:
#         fig, ax = pl.subplots(nrows=2, ncols=1, figsize=(10, 12), dpi=100,sharex=True)
#     else:
#         fig, ax = pl.subplots(figsize=(10, 6), dpi=100)
#         ax=[ax]

        
#     for i in range(len(ax)):
#         ax[i].set_xscale(xscale[i])
#         ax[i].set_yscale(yscale[i])
#         if is_singleplot:
#             ax[0].plot(f[0], f[1], colors[i % 6] + dots(dotted), linestyle=linestyle_str[i % 4],linewidth=line_width)
#             if zoomed:
#                 axins = ax[i].inset_axes(zoomed_position)
#                 axins.plot(f[0], f[1],"b",linewidth=line_width)
#                 axins.set_xlim(zoomed_xlim)
#                 axins.set_ylim(zoomed_ylim)
#                 ax[i].indicate_inset_zoom(axins)
#         elif stacked_plots:
#             for j in range(len(f[i])):
#                 ax[i].plot(f[i][j][0], f[i][j][1], colors[j % 6] + dots(dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
#         else:
#             axins = ax[i].inset_axes(zoomed_position)   if zoomed else None
#             for j in range(len(f)):
#                 ax[i].plot(f[j][0], f[j][1], colors[j % 6] + dots(dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
#                 if zoomed:
#                     axins.plot(f[j][0], f[j][1], colors[j % 6] + dots(zoomed_dotted), linestyle=linestyle_str[j % 4],linewidth=line_width)
#                     axins.set_xlim(zoomed_xlim)
#                     axins.set_ylim(zoomed_ylim)
#                     ax[i].indicate_inset_zoom(axins)
                
                
        
#         ax[i].set_title(title, fontsize=title_fontsize)

#         ax[i].set_xlabel(xlabel[i], fontsize=xy_labels_fontsize)
#         ax[i].set_ylabel(ylabel[i], fontsize=xy_labels_fontsize)
        
#         ax[i].tick_params(axis='both', which='both', labelsize=labelsize)

#         if func_to_compare is not None:
#             x = np.linspace(*ax[i].get_xlim())
#             function = np.vectorize(func_to_compare)
#             ax[i].plot(x, function(x), "r")
        
#         if xlim is not None:
#             ax[i].set_xlim(xlim[i])
#         if ylim is not None:
#             ax[i].set_ylim(ylim[i])
        
#         if len(x_ticks) != 0 and len(x_ticklabels) == len(x_ticks):
#             ax[i].set_xticks(x_ticks)
#             ax[i].set_xticklabels(x_ticklabels)
#         elif len(x_ticks) != 0 and len(x_ticklabels) == 0:
#             ax[i].set_yticks(x_ticks)
#             for ytick in y_ticks:
#                 x_ticklabels.append(str(ytick))
#             ax[i].set_yticklabels(x_ticklabels)
#         if len(y_ticks) != 0 and len(y_ticklabels) == len(y_ticks):
#             ax[i].set_yticks(y_ticks)
#             ax[i].set_yticklabels(y_ticklabels)
#         elif len(y_ticks) != 0 and len(y_ticklabels) == 0:
#             ax[i].set_yticks(y_ticks)
#             for ytick in y_ticks:
#                 y_ticklabels.append(str(ytick))
#             ax[i].set_yticklabels(y_ticklabels)

#         if legend[i]!=None:
#             ax[i].legend(legend[i], shadow=True, loc=location, handlelength=1.5, fontsize=legend_fontsize,ncol=ncol)
               
#     pl.tight_layout()
    
#     if save==True:
#         dpi=80
#         now = datetime.now()
#         current_time = now.strftime("%Y_%m")
#         if name!=None:
#             pl.savefig(save_plot_dir+"/"+name+"_"+current_time+format_,dpi=dpi)
#         else:
#             pl.savefig(save_plot_dir+"/"+"_"+current_time+format_,dpi=dpi)

#     pl.show()

#     return None
