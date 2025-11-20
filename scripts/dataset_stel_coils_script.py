from time import time
import jax.numpy as jnp
import pyvista as pv
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import Tracing
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coils_for_nearaxis
import requests
from essos.coils import Curves


# Other  Near-Axis field parameters
nphi_internal_pyQSC = 51
r_surface = 0.1
ntheta = 41
r_coils=0.4
ncoils=2

#Running pyQSC to retrieve configurartion geometry
