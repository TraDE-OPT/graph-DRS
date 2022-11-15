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
This file contains the implementations of the optimization methods considered in
Section 5.2 in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.

The implementation of the graph-DRS method (cf., svm_gdrs), has been carried out
with a simple size-reduction trick leading to an equivalent method, which, however,
requires storing significantly less variables. Specifically, for every edge (i,j),
with i being an official and j being an agent, one can note that the update on
w_{(i,j)}^{k+1} can be written as

w_{(i,j)}^{k+1} = x_i^{k+1} + k_h*z_{(i,j)}^{k+1}

where h is the index corresponding to the jth agent and z_{(i,j)}^{k+1} is a
number. Thus, instead of storing the variables w_{(i,j)}, which are of size n,
we consider the variables z_{(i,j)}, which are of size 1, for every
aforementioned edge (i,j). Please note that this does not affect the results.
"""


import numpy as np
import scipy.sparse as sp
import structures as st


def svm_gdrs(Data, Labels, Off, maxit, tau):

    n = len(Data)  # number of agents
    k = len(Off)   # number of officials
    p = int(n/k)   # number of agents per officials
    n_nodes = n+k  # number of nodes

    sig = 0.2     # kernel variance
    gamma = 0.01  # Hilbert space norm weight

    # initialize variables
    Ones = np.ones(n)
    x = np.zeros((n, k))    # solution estimates from officials
    z = np.zeros((p, k))    # reduced w variables assigned to agents
    #                         (reduction from size n to size 1 for all agents)
    w = np.zeros((n, k-1))  # w variables relative to edges between consecutive officials

    # convergence metrics
    fobj = []
    var = []

    degs = [p+2 for _ in range(k)]                                       # degrees of officials
    K, prox_tau_K = st.create_K(tau*gamma/sum(degs), Data, sig,
                                    st.ker)  # proximity operator of officials

    y = np.array(Labels)
    K_t = -np.multiply(y, K.T).T  # rows of K times corresponding label y

    d = [np.square(np.linalg.norm(K[c*p:(c+1)*p, :], axis=1))
             for c in range(k)]  # norms of rows of K (= norms rows of K_t)

    for _it in range(maxit):

        z_old = np.copy(z)
        x_old = np.copy(x)
        x_temp = np.copy(x[:, 0])
        x[:, 0] = prox_tau_K@(p/(p+2)*x[:, 0]+((K_t[:p, :]).T@z[:, 0]+w[:, 0])/(p+2))
        z[:, 0] = st.prox_local(tau, z[:, 0], d[0], x[:, 0], x_temp, K_t[:p, :], st.soft, p)

        for c in range(1, k-1):

            x_temp = np.copy(x[:, c])
            x[:, c] = prox_tau_K@(2/(p+2)*x[:, c-1]+p/(p+2)*x[:, c]\
                                      + ((K_t[c*p:(c+1)*p, :]).T@z[:, c]+w[:, c]-w[:, c-1])/(p+2))
            z[:, c] = st.prox_local(tau, z[:, c], d[c], x[:, c], x_temp,
                                        K_t[c*p:(c+1)*p, :], st.soft, p)

        x_temp = np.copy(x[:, -1])
        x[:, -1] = prox_tau_K@(2/(p+2)*(x[:, 0]+x[:, -2])+p/(p+2)*x[:, -1]\
                                   + ((K_t[-p:, :]).T@z[:, -1]-w[:, -1])/(p+2))
        z[:, -1] = st.prox_local(tau, z[:, -1], d[-1], x[:, -1], x_temp, K_t[-p:, :], st.soft, p)

        for c in range(k-1):
            w[:, c] = w[:, c]+x[:, c+1]-x[:, c]

        Xs = st.create_xs(K_t, x, x_old, z, z_old, n, k, n_nodes, p)

        fobj_val, var_val = st.fobj_and_var(Xs, gamma, K_t, K, Ones, n_nodes)
        fobj.append(fobj_val)
        var.append(var_val)

    return Xs, np.array(fobj), np.array(var)


def svm_pdhg(Data, Labels, Off, maxit, tau):

    n = len(Data)
    n_caps = len(Off)
    n_nodes = n+n_caps
    p = int(n/n_caps)

    L = st.create_L(n_caps, n_nodes, p)
    norm_L = np.linalg.norm(L.todense(), 2)
    sigma = 1/(tau*(norm_L**2))
    sig = 0.2     # kernel variance
    gamma = 0.01  # Hilbert space norm weight

    # structures
    Offs = [(p+1)*c for c in range(n_caps)]

    def compl(i):
        return not i % (p+1) == 0

    Offs_comp = [compl(i) for i in range(n+n_caps)]

    Ones = np.ones(n)
    degs = [(p+2) for _ in range(n_caps)]
    K, prox_tau_K = st.create_K(gamma*tau*(p+2)/sum(degs), Data, sig, st.ker)
    K_t = -np.multiply(np.array(Labels), K.T).T

    d = np.square(np.linalg.norm(K_t, axis=1))

    x = np.zeros((n, n_nodes))
    y = np.zeros((n, n_nodes))

    fobj = []
    var = []

    for _k in range(maxit):

        x_old = np.copy(x)
        x = st.big_prox(tau, x-tau*y@L, prox_tau_K, K_t, d, Ones, Offs, Offs_comp)
        xi = 2*x-x_old
        y = y+sigma*xi@L

        fobj_val, var_val = st.fobj_and_var(x, gamma, K_t, K, Ones, n_nodes)
        fobj.append(fobj_val)
        var.append(var_val)

    return x, fobj, var


def svm_pxtr(Data, Labels, Off, maxit, tau):

    n = len(Data)
    k = len(Off)
    n_nodes = n+k
    p = int(n/k)
    sig = 0.2     # kernel variance
    gamma = 0.01  # Hilbert space norm weight

    # convergence metrics
    fobj = []
    var = []

    # structures
    Offs = [(p+1)*c for c in range(k)]

    def compl(i):
        return not i % (p+1) == 0

    Offs_comp = [compl(i) for i in range(n+k)]

    Ones = np.ones(n)
    degs = [(p+2) for _ in range(k)]
    K, prox_tau_K = st.create_K(tau*gamma*(p+2)/sum(degs), Data, sig, st.ker)
    K_t = -np.multiply(np.array(Labels), K.T).T

    # mixing matrices
    W = st.create_W(k, n_nodes, p)
    W_t = (W+sp.eye(np.shape(W)[0]))/2

    d = np.square(np.linalg.norm(K_t, axis=1))

    x_old = np.zeros((n, n_nodes))
    y = x_old@W
    x = st.big_prox(tau, y, prox_tau_K, K_t, d, Ones, Offs, Offs_comp)

    for _it in range(maxit):

        y = x@W+y-x_old@W_t
        x_old = np.copy(x)
        x = st.big_prox(tau, y, prox_tau_K, K_t, d, Ones, Offs, Offs_comp)

        fobj_val, var_val = st.fobj_and_var(x, gamma, K_t, K, Ones, n_nodes)
        fobj.append(fobj_val)
        var.append(var_val)

    return x, fobj, var
