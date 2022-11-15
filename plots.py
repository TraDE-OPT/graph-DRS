# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 Kristian Bredies (kristian.bredies@uni-graz.at)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Emanuele Naldi (e.naldi@tu-braunschweig.de)
#
#    This file is part of the example code repository for the paper:
#
#      K. Bredies, E. Chenchene, E. Naldi.
#      Graph and distributed extensions of the Douglas-Rachford method,
#      2022. DOI: 10.48550/arXiv.2211.04782
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains useful functions to plot our numerical results.
For details and references, see Section 5 in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.
"""

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import data as dt


def plot_experiment_connectivity(Res_list, Cons, title):

    plt.figure()

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
    rc('text', usetex=True)

    for i in range(38):

        conn = Cons[i]
        if conn < 3.8:
            plt.semilogy(Res_list[:, i], color=(conn/2.001, 1-conn/2.001, 1-conn/2.001), alpha=0.5)
        else:
            plt.semilogy(Res_list[:, i], 'k', linewidth=3)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=r'$\lambda_1 = 4$',
                                  markerfacecolor='k', markersize=12),
                       Line2D([0], [0], marker='o', color='w', label=r'$\lambda_1 = 2$',
                                  markerfacecolor=(1, 0, 0), markersize=12),
                       Line2D([0], [0], marker='o', color='w', label=r'$\lambda_1 = 1$',
                                  markerfacecolor=(1/2, 1/2, 1/2), markersize=12),
                       Line2D([0], [0], marker='o', color='w', label=r'$\lambda_1 = 0.5857...$',
                                  markerfacecolor=((2-np.sqrt(2))/2.001, 1-(2-np.sqrt(2))/2.001,
                                                       1-(2-np.sqrt(2))/2.001), markersize=12)]

    plt.xlabel('Iterations')
    plt.ylabel('State variance')
    plt.xlim([0, np.shape(Res_list)[0]])
    plt.legend(handles=legend_elements, prop={'size': 10})
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    plt.savefig(f'Figures/{title}.pdf', bbox_inches='tight')


def plot_experiment_output(sig_0, sig_1, sig_2, sig_3, p, mu, nu, Brg, Wtr):

    Wtr_b = dt.read_image("Data/water_b.png", p)
    Wtr_b = Wtr_b.astype("float64")
    Wtr_b[Wtr_b < 1e-5] = np.nan

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
    rc('text', usetex=True)

    mu_plot = np.copy(mu)
    nu_plot = np.copy(nu)
    Brg_plot = np.copy(Brg)*1
    Wtr_plot = np.copy(Wtr)*1
    Brg_plot = Brg_plot.astype("float64")
    Wtr_plot = Wtr_plot.astype("float64")

    mu_plot[mu_plot < 1e-5] = np.nan
    nu_plot[nu_plot < 1e-5] = np.nan
    Brg_plot[Brg_plot < 0.5] = np.nan
    Wtr_plot[Wtr_plot < 0.5] = np.nan

    Sigs = [np.copy(sig_0), np.copy(sig_1), np.copy(sig_2), np.copy(sig_3)]

    Grid = []
    for i in range(p):
        for j in range(p):
            Grid.append((i, j))

    # Zoom
    Zoom = dt.read_image("Data/zoom_section.png", p)
    Zoom = Zoom > 0.5
    Grid_Zoom = np.array(Grid)[Zoom]

    # Rectangle
    min0 = min(Grid_Zoom[:, 0])
    max0 = max(Grid_Zoom[:, 0])
    min1 = min(Grid_Zoom[:, 1])
    max1 = max(Grid_Zoom[:, 1])

    # lenghts
    lx = max0-min0+1
    ly = max1-min1+1

    X, Y = np.meshgrid(np.array(list(range(p))), np.array(list(range(p))))

    _fig, axs = plt.subplots(2, 2, figsize=(55, 55))

    counter = 0
    for i in range(2):
        for j in range(2):

            density = np.linalg.norm(Sigs[counter], axis=1)

            axs[i, j].imshow(np.reshape(Brg_plot, (p, p)).T, cmap='Greens',
                            vmin=0, vmax=0.5, alpha=0.8)
            axs[i, j].imshow(np.reshape(mu_plot, (p, p)).T, cmap='Purples',
                            alpha=1)
            axs[i, j].imshow(np.reshape(nu_plot, (p, p)).T, cmap='Reds',
                            alpha=1)
            axs[i, j].imshow(np.reshape(density, (p, p)).T, cmap='gray_r',
                            alpha=0.7, vmin=0, vmax=0.012)
            axs[i, j].imshow(np.reshape(Wtr_plot, (p, p)).T, cmap="Blues",
                            vmin=0, vmax=0.5, alpha=0.5)
            axs[i, j].imshow(np.reshape(Wtr_b, (p, p)).T, cmap='Greys',
                            vmin=0, vmax=0.5, alpha=0.8)
            axs[i, j].add_patch(Rectangle((min0, min1), lx, ly, linewidth=3,
                                         edgecolor='r', facecolor='none'))
            axs[i, j].contour(X, Y, np.reshape(mu_plot, (p, p)).T, colors="purple")
            axs[i, j].contour(X, Y, np.reshape(nu_plot, (p, p)).T, colors="red")
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            counter += 1

    plt.savefig('Figures/flows_comparisons_early.pdf', bbox_inches='tight')


def plot_zoom(sig_0, sig_1, sig_2, sig_3, p, mu, nu):

    Sig_zoom = [np.copy(sig_0), np.copy(sig_1), np.copy(sig_2), np.copy(sig_3)]

    Grid = []
    for i in range(p):
        for j in range(p):
            Grid.append((i, j))

    # reading zoom area
    Zoom = dt.read_image("Data/zoom_section.png", p)
    Zoom = Zoom > 0.5
    Grid_Zoom = np.array(Grid)[Zoom]

    # building rectangle
    min0 = min(Grid_Zoom[:, 0])
    max0 = max(Grid_Zoom[:, 0])
    min1 = min(Grid_Zoom[:, 1])
    max1 = max(Grid_Zoom[:, 1])

    # lenghts
    lx = max0-min0+1
    ly = max1-min1+1

    Grid_Zoom_sig = []
    for i in range(lx):
        for j in range(ly):
            Grid_Zoom_sig.append((i, j))
    Grid_Zoom_sig = np.array(Grid_Zoom_sig)

    Grid_Zoom_sig_Active = []
    for i in range(lx):
        for j in range(ly):
            if i % 4 == 0 and j % 4 == 0:
                Grid_Zoom_sig_Active.append(True)
            else:
                Grid_Zoom_sig_Active.append(False)

    _fig, axs = plt.subplots(2, 2, figsize=(55, 55))

    counter = 0
    for i in range(2):
        for j in range(2):

            sig_plot_k = np.copy(Sig_zoom[counter][Zoom, :])
            density = np.linalg.norm(sig_plot_k, axis=1)

            axs[i, j].imshow(np.reshape(density, (lx, ly)).T, cmap='binary', alpha=0.5)
            axs[i, j].quiver([p[0] for p in Grid_Zoom_sig[Grid_Zoom_sig_Active]],
                        [p[1] for p in Grid_Zoom_sig[Grid_Zoom_sig_Active]],
                        list(sig_plot_k[Grid_Zoom_sig_Active, 0]),
                        list(-sig_plot_k[Grid_Zoom_sig_Active, 1]),
                        color='black', width=0.003, alpha=1, scale=0.05)
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])

            counter += 1

    plt.savefig('Figures/flows_zoom_early.pdf', bbox_inches='tight')


def plot_final(sig, p, mu, nu, Brg, Wtr):

    Wtr_b = dt.read_image("Data/water_b.png", p)
    Wtr_b = Wtr_b.astype("float64")
    Wtr_b[Wtr_b < 1e-5] = np.nan

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 45})
    rc('text', usetex=True)

    mu_plot = np.copy(mu)
    nu_plot = np.copy(nu)
    Brg_plot = np.copy(Brg)*1
    Wtr_plot = np.copy(Wtr)*1
    Brg_plot = Brg_plot.astype("float64")
    Wtr_plot = Wtr_plot.astype("float64")

    mu_plot[mu_plot < 1e-5] = np.nan
    nu_plot[nu_plot < 1e-5] = np.nan
    Brg_plot[Brg_plot < 0.5] = np.nan
    Wtr_plot[Wtr_plot < 0.5] = np.nan

    Grid = []
    for i in range(p):
        for j in range(p):
            Grid.append((i, j))

    plt.figure(figsize=(55, 55))

    legend_elements_1 = [Line2D([0], [0], marker='o', color='w', label=r'$\mu$',
                                    markerfacecolor=(120/255, 88/255, 152/255), markersize=95),
                       Line2D([0], [0], marker='o', color='w', label=r'$\nu$',
                                    markerfacecolor='r', markersize=95),
                       Line2D([0], [0], marker='o', color='w', label=r'$|\sigma|$',
                                  markerfacecolor="gray", markersize=95)]

    legend_elements_2 = [Line2D([0], [0], marker='s', color='w', label=r'\texttt{Wtr}',
                                    markerfacecolor=(130/255, 150/255, 180/255), markersize=95),
                         Line2D([0], [0], marker='s', color='w', label=r'\texttt{Brg}',
                                    markerfacecolor=(112/255, 168/255, 131/255), markersize=95)]

    legend_1 = plt.legend(handles=legend_elements_1, prop={'size': 120}, loc=1)
    plt.legend(handles=legend_elements_2, prop={'size': 120}, loc=3)
    plt.gca().add_artist(legend_1)

    density = np.linalg.norm(sig, axis=1)
    plt.imshow(np.reshape(Brg_plot, (p, p)).T, cmap='Greens',
               vmin=0, vmax=0.5, alpha=0.5)
    plt.imshow(np.reshape(mu_plot, (p, p)).T, cmap='Purples',
               alpha=1)
    plt.imshow(np.reshape(nu_plot, (p, p)).T, cmap='Reds',
               alpha=1)
    plt.imshow(np.reshape(density, (p, p)).T, cmap='gray_r',
               alpha=0.7, vmin=0, vmax=0.012)
    plt.imshow(np.reshape(Wtr_plot, (p, p)).T, cmap='Blues',
               vmin=0, vmax=0.5, alpha=0.5)
    plt.imshow(np.reshape(Wtr_b, (p, p)).T, cmap='Greys',
               vmin=0, vmax=0.5, alpha=0.8)
    plt.tick_params(labelsize=120)

    X, Y = np.meshgrid(np.array(list(range(p))), np.array(list(range(p))))
    plt.contour(X, Y, np.reshape(mu_plot, (p, p)).T, colors="purple")
    plt.contour(X, Y, np.reshape(nu_plot, (p, p)).T, colors="red")

    plt.savefig('Figures/flow_late.pdf', bbox_inches='tight')


def plot_svm(fobj_gdrs, fobj_pxtr, fobj_pdhg, var_gdrs, var_pxtr, var_pdhg):

    plt.figure()

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
    rc('text', usetex=True)

    # plotting objective function
    # gDRS
    plt.plot(np.mean(fobj_gdrs, axis=1), '-*', color='green',
             linewidth=1, markersize=6, markevery=1000, label='DRS')
    plt.fill_between(range(len(fobj_gdrs)), fobj_gdrs.min(1),
                     fobj_gdrs.max(1), color='green', alpha=0.1)

    # PE
    plt.plot(np.mean(fobj_pxtr, axis=1), '-o', color='red',
             linewidth=1, markersize=6, markevery=1000, label='P-EXTRA')
    plt.fill_between(range(len(fobj_pxtr)), fobj_pxtr.min(1),
                     fobj_pxtr.max(1), color='red', alpha=0.1)

    # CP
    plt.plot(np.mean(fobj_pdhg, axis=1), '-v', color='blue',
             linewidth=1, markersize=6, markevery=1000, label='PDHG')
    plt.fill_between(range(len(fobj_pdhg)), fobj_pdhg.min(1),
                     fobj_pdhg.max(1), color='blue', alpha=0.1)

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Objective function')
    plt.legend(prop={'size': 13})

    plt.savefig('Figures/fobj.pdf', bbox_inches='tight')
    plt.show()

    # plotting state variance

    plt.figure()

    # gDRS
    plt.semilogy(np.mean(var_gdrs, axis=1), '-*', color='green',
                 linewidth=1, markersize=6, markevery=1000, label='DRS')
    plt.fill_between(range(len(var_gdrs)), var_gdrs.min(1),
                     var_gdrs.max(1), color='green', alpha=0.1)
    # PE
    plt.semilogy(np.mean(var_pxtr, axis=1), '-o', color='red',
                 linewidth=1, markersize=6, markevery=1000, label='P-EXTRA')
    plt.fill_between(range(len(var_pxtr)), var_pxtr.min(1),
                     var_pxtr.max(1), color='red', alpha=0.1)
    # CP
    plt.semilogy(np.mean(var_pdhg, axis=1), '-v', color='blue',
                 linewidth=1, markersize=6, markevery=1000, label='PDHG')
    plt.fill_between(range(len(var_pdhg)), var_pdhg.min(1),
                     var_pdhg.max(1), color='blue', alpha=0.1)

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('State variance')

    plt.legend(prop={'size': 13})

    plt.savefig('Figures/var.pdf', bbox_inches='tight')
