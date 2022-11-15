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
Run this script to reproduce all the numerical experiments in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.
"""


import pathlib
import data as dt
from structures import generate_data
from experiments import (compare_methods, experiment_connectivity,
                         experiment_outputs)


def run_experiment_connectivity():

    p = 35
    maxit = 1000

    [mu, nu] = dt.read_measures("Data/mu.png", "Data/nu.png", p)
    Brg, Wtr = dt.read_brg_and_wtr(p)

    experiment_connectivity(mu, nu, Brg, Wtr, maxit)


def run_experiment_output():

    p = 720
    maxit = 300

    [mu, nu] = dt.read_measures("Data/mu.png", "Data/nu.png", p)
    Brg, Wtr = dt.read_brg_and_wtr(p)

    experiment_outputs(mu, nu, Brg, Wtr, maxit)


def run_experiment_svm():

    p = 10  # number of agents per official
    k = 5   # number of officials
    maxit = 10000

    Cap, Data, Labels = generate_data(p, k)
    compare_methods(Data, Labels, Cap, maxit)


if __name__ == "__main__":

    pathlib.Path("Figures").mkdir(parents=True, exist_ok=True)

    print("---------------------------------------------------------------------------")
    print("\n\n *** Starting experiment on the influence of the graph topology ***\n")
    run_experiment_connectivity()

    print("---------------------------------------------------------------------------")
    print("\n\n *** Starting comparisons between different solution estimates ***\n")
    run_experiment_output()

    print("---------------------------------------------------------------------------")
    print("\n\n *** Starting comparisons for the distributed SVM problem ***\n")
    run_experiment_svm()
