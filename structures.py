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
This file contains several useful functions to reproduce the experiments in Section 5 in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.
"""


import random
import numpy as np
import scipy.sparse as sp
import networkx as nx


def compute_onto_decomposition(L):

    d, U = np.linalg.eigh(L)

    Keep = [i > 1e-10 for i in d]
    d_r = d[Keep]

    D = np.diag(d_r)
    U = U[:, Keep]
    Z = U@(np.sqrt(D))

    return Z, d[1]


def create_sparse_gradx_mat(p):

    diag = np.ones(p)
    diag[-1] = 0
    diag = np.tile(diag, p)

    Dx = sp.spdiags([-diag, [0]+list(diag[:-1])], [0, 1], p**2, p**2)

    return Dx


def create_sparse_grady_mat(p):

    diag = np.ones(p**2)
    diag[-p:] = 0*diag[-p:]

    up_diag = np.ones(p**2)
    up_diag[:p] = 0*up_diag[:p]

    Dy = sp.spdiags([-diag, up_diag], [0, p], p**2, p**2)

    return Dy


def create_gradx_mat(p):

    Dx = np.zeros((p**2, p**2))
    Dx_mini = np.zeros((p, p))

    for i in range(p-1):
        Dx_mini[i, i] = -1
        Dx_mini[i, i+1] = 1

    for i in range(p):
        Dx[i*p:(i+1)*p, i*p:(i+1)*p] = Dx_mini

    return Dx


def create_grady_mat(p):

    Id = np.eye(p)
    Dy = np.zeros((p**2, p**2))
    Dy_mini = np.hstack((-Id, Id))

    for i in range(p-1):
        Dy[i*p:(i+1)*p, i*p:(i+2)*p] = Dy_mini

    return Dy


def grad(psi, Dx, Dy, n):

    sig_out = np.zeros((n, 2))

    sig_out[:, 0] = Dx @ psi
    sig_out[:, 1] = Dy @ psi

    return sig_out


def div(sig, M1, M2):

    return M1 @ sig[:, 0] + M2 @ sig[:, 1]


def proj_inf_2(sig, n):

    sig_out = np.copy(sig)
    Norm = np.linalg.norm(sig_out, axis=1)
    Greater = Norm > 1
    sig_out[Greater] = np.divide(sig_out[Greater], np.transpose(np.asmatrix(Norm[Greater])))

    return sig_out


def prox_l1(tau, sig, n, Brg, Wtr, C):

    return sig-tau*proj_inf_2(sig/tau, n)


def prox_l32(tau, sig, n, Brg, Wtr, C):

    norms = np.linalg.norm(sig, axis=1)

    norms_out = norms+9/8*tau**2*(1-np.sqrt(1+16/(9*tau**2)*norms))

    non_zero = norms > 0

    sig_out = np.copy(sig)
    sig_out[non_zero, :] *= (norms_out[non_zero]/norms[non_zero])[:, np.newaxis]

    return sig_out


def prox_bridge(tau, sig, n, Brg, Wtr, C):

    sig_out = np.copy(sig)

    sig_b = np.copy(sig[Brg, :])

    norms_b = np.linalg.norm(sig_b, axis=1)
    Greater = norms_b >= C

    sig_b[Greater] =  C*np.divide(sig_b[Greater], np.transpose(np.asmatrix(norms_b[Greater])))

    sig_out[Brg, :] = sig_b
    sig_out[Wtr, :] = 0*sig_out[Wtr, :]

    return sig_out


def create_xs(K_t, x_c, x_c_old, z_new, z_old, n, k, n_nodes, p):

    Xs = np.zeros((n, n_nodes))
    diff_x = 2*x_c-x_c_old

    for j in range(k):
        Xs[:, j*(p+1)] = x_c[:, j]
        diff_repeated = np.repeat(diff_x[:, j][:, np.newaxis], p, axis=1)
        Xs[:, j*(p+1)+1:j*(p+1)+(p+1)] = diff_repeated + np.multiply(K_t[j*p:(j+1)*p, :].T,
                                                                         z_new[:, j]-z_old[:, j])

    return Xs


def fobj_and_var(Xs, gamma, K_t, K, Ones, n_nodes):

    x_avg = np.mean(Xs, axis=1)
    fobj = sum(np.maximum(K_t@x_avg+Ones, 0))+gamma*x_avg.T@K@x_avg
    var = sum(np.linalg.norm(Xs-np.repeat(x_avg[:, np.newaxis], n_nodes, axis=1), axis=0)**2)

    return fobj, var


def generate_one_label():

    return -1 if random.random() < 0.8 else 1


def generate_data(p, k):

    # officials
    np.random.seed(15)
    Off = np.zeros((k, 2))
    for c in range(k):
        Off[c, :] = np.array([np.cos(2*np.pi*c/k), np.sin(2*np.pi*c/k)]) \
          + np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), 1)
    Labels = []
    Data = []
    Counter = 0

    for c in Off:

        if Counter <= int(k/2-1):
            Labels_c = [generate_one_label() for k in range(p)]
        else:
            Labels_c = [-1*generate_one_label() for k in range(p)]

        G_c = np.random.multivariate_normal(c, 0.1*np.eye(2), p)
        G_c = [np.array([c[0], c[1]]) for c in G_c]
        Data = Data+G_c
        Labels = Labels + Labels_c
        Counter += 1

    return Off, Data, Labels


def soft(xi, tau):

    return np.minimum(xi, 0)+np.maximum(0, xi-tau)


def ker(x, y, sig):

    return np.exp(-np.linalg.norm(x-y)**2/(2*sig))


def create_K(par, Data, sig, kernel):

    n = len(Data)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(Data[i], Data[j], sig)

    prox_tau_K = np.linalg.inv(2*par*K+np.eye(n))

    return K, prox_tau_K


def prox_local(tau, z_c, d_c, x_c, x_c_old, K_c, soft_t, p):

    xi = np.divide(K_c@(2*x_c-x_c_old)+np.ones(p), d_c)-z_c

    return soft_t(xi, tau)-xi


def prox_fidelity_pxtr(tau, x, K_t, d, Ones, soft_t):

    xi = np.divide(np.sum(K_t*x.T, axis=1)+Ones, d)

    return x+K_t.T@np.diag(soft_t(xi, tau)-xi)


def create_W(k, n_nodes, p):

    G = nx.Graph()

    # adding officials
    for c in range(k):
        edges_local = [(c, i+p*c+k) for i in range(p)]
        G.add_edges_from(edges_local)

    Off_edges = [(i, i+1) for i in range(k-1)] + [(0, k-1)]
    G.add_edges_from(Off_edges)

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)

    W = sp.eye(np.shape(L)[0])-L/n_nodes

    return W


def create_L(k, n_nodes, p):

    G = nx.Graph()

    # adding officials
    for c in range(k):
        edges_local = [(c, i+p*c+k) for i in range(p)]
        G.add_edges_from(edges_local)

    Off_edges = [(i, i+1) for i in range(k-1)] + [(0, k-1)]
    G.add_edges_from(Off_edges)

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)

    return L


def big_prox(tau, x_in, prox_tau_K, K_t, d, Ones, Offs, Offs_comp):

    x_out = np.copy(x_in)

    x_out[:, Offs] = prox_tau_K@x_in[:, Offs]
    x_out[:, Offs_comp] = prox_fidelity_pxtr(tau, x_in[:, Offs_comp], K_t, d, Ones, soft)

    return x_out
