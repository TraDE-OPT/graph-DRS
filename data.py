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
Data processing.
For details and references, see Section 5 in:

K. Bredies, E. Chenchene, E. Naldi.
Graph and distributed extensions of the Douglas-Rachford method,
2022. DOI: 10.48550/arXiv.2211.04782.
"""


import numpy as np
from PIL import Image


def read_image(img1, p):

    img_brg = Image.open(img1).convert('L')
    Img = 255-np.array(img_brg.resize((p, p)))
    Img = np.reshape(Img, p**2)

    return Img


def read_measures(img1, img2, p):

    mu = read_image(img1, p)
    mu = mu/np.sum(mu)
    nu = read_image(img2, p)
    nu = nu/np.sum(nu)

    return mu, nu


def read_brg_and_wtr(p=10):

    # reading Brg
    Brg = read_image("Data/bridge.png", p)
    Brg = Brg > 0.5

    # reading Wtr
    Wtr = read_image("Data/water.png", p)
    Wtr = Wtr > 0.5

    # check that there are no overlaps
    for i in range(p**2):
        if Wtr[i] and Brg[i]:
            Brg[i] = False

    return list(Brg), list(Wtr)
