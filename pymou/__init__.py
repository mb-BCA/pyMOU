#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, Andrea Insabato, Gorka Zamora-López, Matthieu Gilson
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Multivariate Ornstein-Uhlenbeck process
=======================================

A package to simulate the Multivariate Ornstein-Uhlenbeck (MOU) process and 
fit it to time series (i.e. optimize its parameters like connectivity).

An application is the estimation of whole-brain effective connectivity for fMRI
data with parcellation of ~100 regions of interest (ROIs).

The package consists of 2 Modules:

mou_model
    Core module. Defines class MOU.
tools
    Diverse helper functions (e.g. to generate a random connectivity matrix).

References:
- Gilson M, Moreno-Bote R, Ponce-Alvarez A, Ritter P, Deco G (2016) "Estimation
of directed Effective Connectivity from fMRI Functional Connectivity Hints at
Asymmetries of Cortical Connectome" PLoS Comput Biol 12: e1004762, 
https://doi.org/10.1371/journal.pcbi.1004762.
- Gilson M, Deco G, Friston K, Hagmann P, Mantini D, Betti V, Romani GL, 
Corbetta M (2018) "Effective connectivity inferred from fMRI transition 
dynamics during movie viewing points to a balanced reconfiguration of cortical
interactions" Neuroimage 180: 534-546, 
https://doi.org/10.1016/j.neuroimage.2017.09.061;
- Gilson M, Zamora-López G, Pallarés V, Adhikari MH, Senden M, Tauste Campo A, 
Mantini D, Corbetta M, Deco G, Insabato A (biorxiv) "MOU-EC: model-based 
whole-brain effective connectivity to extract biomarkers for brain dynamics 
from fMRI data and study distributed cognition", http://doi.org/10.1101/531830;


Using pyMOU
-----------

The notebook 'MOU_Simulation_Estimation.ipynb' in the directory 'examples/' 
illustrates how to use the MOU class for simulation of the MOU process and
estimation of the connectivity from observed activity.


Further information
-------------------

To see the list of all functions available within each module, use the
standard help in an interactive session, e.g.,  ::

>>> import mou_model
>>> help(mou_model)

Same, to find further details of every function within each module:, e.g.,  ::

>>> help(mou_model.MOU)


License
-------

See LICENSE.txt file.
Copyright (C) 2019 - Andrea Insabato, Gorka Zamora-López, Matthieu Gilson

"""
from __future__ import absolute_import

from . import mou_model
from .mou_model import *

__author__ = "Andrea Insabato, Gorka Zamora-Lopez and Matthieu Gilson"
__email__ = "galib@Zamora-Lopez.xyz"
__copyright__ = "Copyright 2019"
__license__ = "Apache License version 2.0"
__version__ = "0.1.0"
__update__="01/11/2019"