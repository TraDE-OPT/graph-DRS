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
This file contains all the experiments in Section 5 in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.
"""


import numpy as np
import scipy.sparse.linalg as spl
import structures as st
import plots as show
from svm_methods import svm_gdrs, svm_pdhg, svm_pxtr


def optimize_exp_1(Z, L, tau, n, mu, nu, proj_div, Brg, Wtr, C, maxit=100):

    # starting point
    w1 = np.zeros((n, 2))
    w2 = np.zeros((n, 2))
    w3 = np.zeros((n, 2))

    Res = []

    for _i in range(maxit):

        sig_0 = proj_div(tau/L[0, 0], (Z[0, 0]*w1+Z[0, 1]*w2+Z[0, 2]*w3)/L[0, 0], n, Brg, Wtr, C)
        sig_1 = st.prox_l32(tau/L[1, 1], -2*L[0, 1]*sig_0/L[1, 1] +
                            (Z[1, 0]*w1+Z[1, 1]*w2+Z[1, 2]*w3)/L[1, 1], n, Brg, Wtr, C)
        sig_2 = st.prox_l1(tau/L[2, 2], -2*(L[0, 2]*sig_0+L[1, 2]*sig_1)/L[2, 2] +
                            (Z[2, 0]*w1+Z[2, 1]*w2+Z[2, 2]*w3)/L[2, 2], n, Brg, Wtr, C)
        sig_3 = st.prox_bridge(tau/L[3, 3], -2*(L[0, 3]*sig_0+L[1, 3]*sig_1+L[2, 3]*sig_2)/L[3, 3] +
                            (Z[3, 0]*w1+Z[3, 1]*w2+Z[3, 2]*w3)/L[3, 3], n, Brg, Wtr, C)

        w1 = w1-(Z[0, 0]*sig_0+Z[1, 0]*sig_1+Z[2, 0]*sig_2+Z[3, 0]*sig_3)
        w2 = w2-(Z[0, 1]*sig_0+Z[1, 1]*sig_1+Z[2, 1]*sig_2+Z[3, 1]*sig_3)
        w3 = w3-(Z[0, 2]*sig_0+Z[1, 2]*sig_1+Z[2, 2]*sig_2+Z[3, 2]*sig_3)

        Mean = 1/4*(sig_0+sig_1+sig_2+sig_3)
        Res.append(np.sqrt(np.linalg.norm(sig_0-Mean)**2 +
                    np.linalg.norm(sig_1-Mean)**2 +
                    np.linalg.norm(sig_2-Mean)**2 +
                    np.linalg.norm(sig_3-Mean)**2))

    return Res


def optimize_exp_2(Z, L, tau, n, mu, nu, proj_div, Brg, Wtr, C, maxit=100):

    # starting point
    w1 = np.zeros((n, 2))
    w2 = np.zeros((n, 2))
    w3 = np.zeros((n, 2))

    Res = []

    for _i in range(maxit):

        sig_0 = proj_div(tau/3, (Z[0, 0]*w1+Z[0, 1]*w2+Z[0, 2]*w3)/3, n, Brg, Wtr, C)
        sig_1 = st.prox_l32(tau/3, 2*sig_0/3 +
                            (Z[1, 0]*w1+Z[1, 1]*w2+Z[1, 2]*w3)/3, n, Brg, Wtr, C)
        sig_2 = st.prox_l1(tau/3, 2*(sig_0+sig_1)/3 +
                            (Z[2, 0]*w1+Z[2, 1]*w2+Z[2, 2]*w3)/3, n, Brg, Wtr, C)
        sig_3 = st.prox_bridge(tau/3, 2*(sig_0+sig_1+sig_2)/3 +
                            (Z[3, 0]*w1+Z[3, 1]*w2+Z[3, 2]*w3)/3, n, Brg, Wtr, C)

        w1 = w1-(Z[0, 0]*sig_0+Z[1, 0]*sig_1+Z[2, 0]*sig_2+Z[3, 0]*sig_3)
        w2 = w2-(Z[0, 1]*sig_0+Z[1, 1]*sig_1+Z[2, 1]*sig_2+Z[3, 1]*sig_3)
        w3 = w3-(Z[0, 2]*sig_0+Z[1, 2]*sig_1+Z[2, 2]*sig_2+Z[3, 2]*sig_3)

        Mean = 1/4*(sig_0+sig_1+sig_2+sig_3)
        Res.append(np.sqrt(np.linalg.norm(sig_0-Mean)**2 +
                           np.linalg.norm(sig_1-Mean)**2 +
                           np.linalg.norm(sig_2-Mean)**2 +
                           np.linalg.norm(sig_3-Mean)**2))

    return Res


def experiment_connectivity(mu, nu, Brg, Wtr, maxit=100):

    n = len(mu)
    p = int(np.sqrt(n))

    tau = 2   # step-size
    C = 0.05  # bridge capacity

    Dy = st.create_gradx_mat(p)  # Switched because of row-wise stacking of the pictures
    Dx = st.create_grady_mat(p)

    M1 = -Dx.T
    M2 = -Dy.T
    Lap = -(Dx.T@Dx+Dy.T@Dy)
    pinv_Lap = np.linalg.pinv(Lap)

    def proj_div(tau, sig, n, Brg, Wtr, C):
        return sig-st.grad(pinv_Lap@(st.div(sig, M1, M2)+mu-nu), Dx, Dy, n)

    Res_list_1 = np.zeros((maxit, 38))
    Res_list_2 = np.zeros((maxit, 38))
    Cons = np.zeros(38)

    i = 0
    for l01 in [-1, 0]:
        for l02 in [-1, 0]:
            for l03 in [-1, 0]:
                for l12 in [-1, 0]:
                    for l13 in [-1, 0]:
                        for l23 in [-1, 0]:
                            L = np.array([[-l01-l02-l03, l01, l02, l03],
                                          [l01, -l01-l12-l13, l12, l13],
                                          [l02, l12, -l02-l12-l23, l23],
                                          [l03, l13, l23, -l03-l13-l23]])

                            if np.linalg.matrix_rank(L) == 3:

                                if i % 5 == 0:
                                    print(f'Graph number: {i+1}. Remaining graphs: {38-i-1}')

                                Z, conn = st.compute_onto_decomposition(L)

                                Res_exp1 = optimize_exp_1(Z, L, tau, n, mu, nu,
                                                          proj_div, Brg, Wtr, C, maxit)

                                Res_exp2 = optimize_exp_2(Z, L, tau, n, mu, nu,
                                                          proj_div, Brg, Wtr, C, maxit)

                                Cons[i] = conn
                                Res_list_1[:, i] = Res_exp1
                                Res_list_2[:, i] = Res_exp2

                                i = i+1

    # plot the results
    show.plot_experiment_connectivity(Res_list_1, Cons, "experiment_connectivity_1")
    show.plot_experiment_connectivity(Res_list_2, Cons, "experiment_connectivity_2")
    print("\n* Results of experiment connectivity saved in Figures as\n" +
          "---> experiment_connectivity_1.pdf and experiment_connectivity_2.pdf\n")


def optimize_experiment_output(L, tau, p, n, mu, nu, proj_div, Brg, Wtr, C, maxit):

    print("Information displayed every 50 iterations.")
    Z, _conn = st.compute_onto_decomposition(L)

    # initialization
    w1 = np.zeros((n, 2))
    w2 = np.zeros((n, 2))
    w3 = np.zeros((n, 2))

    for i in range(maxit):

        sig_0 = proj_div(tau/L[0, 0], (Z[0, 0]*w1+Z[0, 1]*w2+Z[0, 2]*w3)/L[0, 0], n, Brg, Wtr, C)
        sig_1 = st.prox_l32(1.5*tau/L[1, 1], -2*L[0, 1]*sig_0/L[1, 1] +
                            (Z[1, 0]*w1+Z[1, 1]*w2+Z[1, 2]*w3)/L[1, 1], n, Brg, Wtr, C)
        sig_2 = st.prox_l1(tau/L[2, 2], -2*(L[0, 2]*sig_0+L[1, 2]*sig_1)/L[2, 2] +
                            (Z[2, 0]*w1+Z[2, 1]*w2+Z[2, 2]*w3)/L[2, 2], n, Brg, Wtr, C)
        sig_3 = st.prox_bridge(tau/L[3, 3], -2*(L[0, 3]*sig_0+L[1, 3]*sig_1+L[2, 3]*sig_2)/L[3, 3] +
                            (Z[3, 0]*w1+Z[3, 1]*w2+Z[3, 2]*w3)/L[3, 3], n, Brg, Wtr, C)

        w1 = w1-(Z[0, 0]*sig_0+Z[1, 0]*sig_1+Z[2, 0]*sig_2+Z[3, 0]*sig_3)
        w2 = w2-(Z[0, 1]*sig_0+Z[1, 1]*sig_1+Z[2, 1]*sig_2+Z[3, 1]*sig_3)
        w3 = w3-(Z[0, 2]*sig_0+Z[1, 2]*sig_1+Z[2, 2]*sig_2+Z[3, 2]*sig_3)

        if i == 30:

            # compute variance
            Mean = 1/4*(sig_0+sig_1+sig_2+sig_3)
            var = np.linalg.norm(sig_0-Mean)**2 + np.linalg.norm(sig_1-Mean)**2 + \
                np.linalg.norm(sig_2-Mean)**2 + np.linalg.norm(sig_3-Mean)**2

            # display information
            print(f"Iteration 30 reached. State variance: {var}")

            # plot results
            show.plot_experiment_output(sig_0, sig_1, sig_2, sig_3, p, mu, nu, Brg, Wtr)
            show.plot_zoom(sig_0, sig_1, sig_2, sig_3, p, mu, nu)

            print("\n* Results of early comparisons between different solution " +
                  "estimates saved in Figures as\n" +
                  "---> flows_comparisons_early.pdf\n")
            print("* Zoom of the red squares saved in Figures as\n" +
                  "---> flows_zoom_early.pdf\n")

        if i % 50 == 0 and i > 0:

            # compute variance
            Mean = 1/4*(sig_0+sig_1+sig_2+sig_3)
            var = np.linalg.norm(sig_0-Mean)**2 + np.linalg.norm(sig_1-Mean)**2 + \
                np.linalg.norm(sig_2-Mean)**2 + np.linalg.norm(sig_3-Mean)**2

            # display information
            print(f'Iteration: {i}. State variance: {var}')

    # plot results at iteration maxit
    print(f'Final iteration: {i}. State variance: {var}')
    show.plot_final(sig_2, p, mu, nu, Brg, Wtr)
    print("\n* Solution estimate at iteration 300 saved in Figures as\n" +
            "---> flow_late.pdf\n")


def build_laplacian_full():

    L = -np.ones((4, 4))
    np.fill_diagonal(L, 3)

    return L


def experiment_outputs(mu, nu, Brg, Wtr, maxit):

    n = len(mu)
    p = int(np.sqrt(n))

    tau = 1e-1
    C = 0.01

    Dy = st.create_sparse_gradx_mat(p)
    Dx = st.create_sparse_grady_mat(p)

    M1 = -Dx.T
    M2 = -Dy.T
    Lap = -(Dx.T@Dx+Dy.T@Dy)

    def proj_div(tau, sig, n, Brg, Wtr, C):
        return sig-st.grad(spl.spsolve(Lap,
                             st.div(sig, M1, M2)+mu-nu), Dx, Dy, n)

    L = build_laplacian_full()

    optimize_experiment_output(L, tau, p, n, mu, nu, proj_div, Brg, Wtr, C, maxit)


def compare_methods(Data, Labels, Cap, maxit):

    k_max = 10  # number of independent runs

    v = np.logspace(-2, 1, num=k_max)
    v_gdrs = np.copy(v)
    v_pxtr = np.copy(v)
    v_pdhg = np.copy(v)

    fobj_gdrs = np.zeros((maxit, k_max))
    var_gdrs = np.zeros((maxit, k_max))
    fobj_pxtr = np.zeros((maxit, k_max))
    var_pxtr = np.zeros((maxit, k_max))
    fobj_pdhg = np.zeros((maxit, k_max))
    var_pdhg = np.zeros((maxit, k_max))

    for k in range(k_max):

        print(f'#### Starting run number {k+1}, step-size = {v_gdrs[k]}')
        _x_gdrs, fobj_gdrs[:, k], var_gdrs[:, k] = svm_gdrs(Data, Labels, Cap, maxit, tau=v_gdrs[k])
        _x_pxtr, fobj_pxtr[:, k], var_pxtr[:, k] = svm_pxtr(Data, Labels, Cap, maxit, tau=v_pxtr[k])
        _x_pdhg, fobj_pdhg[:, k], var_pdhg[:, k] = svm_pdhg(Data, Labels, Cap, maxit, tau=v_pdhg[k])

    # plot results
    show.plot_svm(fobj_gdrs, fobj_pxtr, fobj_pdhg, var_gdrs, var_pxtr, var_pdhg)
    print("\n* Result of the distributed SVM experiment saved in Figures as\n" +
          "---> var.pdf and fobj.pdf\n")
