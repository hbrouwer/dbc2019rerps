#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Harm Brouwer <me@hbrouwer.eu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---- Last modified: September 2020, Harm Brouwer ----

"""regression-based ERP estimation.
    
Minimal implementation of regression-based ERP (rERP) waveform estimation,
as proposed in:

Smith, N.J., Kutas, M., Regression-based estimation of ERP waveforms: I. The
    rERP framework, Psychophysiology, 2015, Vol. 52, pp. 157-168

Smith, N.J., Kutas, M., Regression-based estimation of ERP waveforms: II.
    Non-linear effects, overlap correction, and practical considerations,
    Psychophysiology, 2015, Vol. 52, pp. 169-181

"""
__all__ = ["models", "plots"]
