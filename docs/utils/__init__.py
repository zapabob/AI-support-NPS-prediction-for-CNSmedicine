"""
Compound Activity Predictor Package

This package provides tools for predicting the activity of chemical compounds
using machine learning models such as Graph Convolutional Networks (GCNs) and
Random Forests.

Modules:
    data_utils: Utilities for loading and preprocessing data.
    descriptor_utils: Utilities for calculating molecular descriptors.
    model_utils: Utilities for training and evaluating machine learning models.
    quantum_utils: Utilities for calculating quantum chemical features.
"""

import os
import sys
import traceback

# Try importing required modules
try:
    import numpy as np
    import pandas as pd
    import dgl
    import torch
    from rdkit import Chem
except ImportError as e:
    print(f"Error importing required modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try importing package modules
try:
    from .data_utils import *
    from .descriptor_utils import *
    from .model_utils import *
    from .quantum_utils import *
except ImportError as e:
    print(f"Error importing package modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Define package metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__version__ = "0.1"
