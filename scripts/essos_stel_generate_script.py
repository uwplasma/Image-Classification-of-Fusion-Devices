'''
This generates stelarator images with coils
'''

import os
import random
from collections import defaultdict
from time import time
import pandas as pd
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

def create_stel(rc1, rc2, rc3, zs1, zs2, zs3, etabar, nfp):
    rc= [1, rc1, rc2, rc3]
    zs= [0, zs1, zs2, zs3]
    int_nfp = int(nfp)
    return near_axis(rc=rc, zs=zs, etabar=etabar, nfp=int_nfp, nphi=nphi_internal_pyQSC)

def generate_stel_image(nfp, cmap, coil_color, field_nearaxis, filename):
    #current_on_each_coil = current_on_each_coil / ncoils*r_surface**2/1.7**2
    n_segments = ntheta
    stellsym = True
    n_curves = ncoils

    # Optimization parameters
    max_coil_length = 6
    max_coil_curvature = 6
    order_Fourier_series_coils = 5
    order=5
    number_coil_points = order_Fourier_series_coils*10
    maximum_function_evaluations = 10
    number_coils_per_half_field_period = ncoils
    tolerance_optimization = 1e-8

    # Initialize coils
    current_on_each_coil = 17e5*field_nearaxis.B0/nfp/2
    #current_on_each_coil = current_on_each_coil / ncoils*r_surface**2/1.7**2
    n_segments = ntheta

    # Initialize coils
    time0 = time()
    number_of_field_periods = nfp
    Raxis = field_nearaxis.R0
    Zaxis = field_nearaxis.Z0
    phi = field_nearaxis.phi
    Xaxis = Raxis*jnp.cos(phi)
    Yaxis = Raxis*jnp.sin(phi)
    X1 = r_coils*jnp.cos(phi)
    Y1 = r_coils*jnp.sin(phi)
    number_coil_points = nphi_internal_pyQSC
    points = jnp.array([Xaxis, Yaxis, Zaxis])


    angles = (jnp.arange(number_coils_per_half_field_period) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)
    curves = jnp.zeros((n_curves, 3, 1 + 2 * order))
    X_value=jnp.interp(angles, phi, Xaxis, period=2*jnp.pi/nfp)
    Y_value=jnp.interp(angles, phi, Yaxis, period=2*jnp.pi/nfp)
    Z_value=jnp.interp(angles, phi, Zaxis, period=2*jnp.pi/nfp)
    X1_value=jnp.interp(angles, phi, X1, period=2*jnp.pi/nfp)
    Y1_value=jnp.interp(angles, phi, Y1, period=2*jnp.pi/nfp)
    curves = curves.at[:, 0, 0].set(X_value)  # x[0]
    #curves = curves.at[:, 0, 1].set(-r_coils)
    curves = curves.at[:, 0, 2].set(X1_value)  # x[2]
    curves = curves.at[:, 1, 0].set(Y_value)  # y[0]
    #curves = curves.at[:, 1, 1].set(-r_coils)
    curves = curves.at[:, 1, 2].set(Y1_value)  # y[2]
    curves = curves.at[:, 2, 0].set(Z_value)  # y[0]
    curves = curves.at[:, 2, 1].set(-r_coils)  # y[2]                   # z[1] (constant for all)

    curves_complete=Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)
    coils_initial = Coils(curves=curves_complete, currents=[current_on_each_coil]*number_coils_per_half_field_period)
    print(f"Creating coils_gamma took {time()-time0:.2f} seconds for {ncoils*2*nfp} coils")


    # Optimize coils
    print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
    time0 = time()
    initial_dofs = coils_initial.x
    coils_optimized = optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=coils_initial.x,
                                    coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                    maximum_function_evaluations=maximum_function_evaluations, field_nearaxis=field_nearaxis,
                                    max_coil_length=max_coil_length, max_coil_curvature=max_coil_curvature)
    print(f"Optimization took {time()-time0:.2f} seconds")

    azim_default = None
    elevParams = [90, 30, -50]
    helicity=0
    if azim_default == None:
        if helicity == 0:
            azim_default = 0
        else:
            azim_default = 45


    # # Save results in vtk format to analyze in Paraview
    # coils_initial.to_vtk('coils_initial')
    coils_optimized.to_vtk('coils_optimized')
    field_nearaxis.to_vtk('nearaxis_field', r=r_surface, ntheta=41, nphi=101)


    coils_opt_file='coils_optimized.vtu'
    field_file='nearaxis_field.vts'


    # Load the coil and field files
    field_mesh = pv.read(field_file)
    coils_mesh = pv.read(coils_opt_file)

    #Function to use with pyvista plotter to change rcamera rotation based on aximuthal angle and elevation angle
    def camera_rotation(pl,azim_angle=0,elev_angle=0):
        cam = pl.camera

        # Load current camera state
        pos = jnp.array(cam.position)
        f = jnp.array(cam.focal_point)
        up = jnp.array(cam.up)

        # Camera vector (from focal point to camera)
        v = pos - f
        r = jnp.linalg.norm(v)

        # Convert to spherical coordinates
        x, y, z = v
        theta = jnp.arctan2(y, x)        # azimuth angle
        phi   = jnp.arccos(z / r)        # polar angle

        # Apply changes (in radians)
        theta += jnp.radians(azim_angle)
        phi   += jnp.radians(elev_angle)

        # Clamp phi to avoid flipping over the poles
        phi = jnp.clip(phi, 1e-3, jnp.pi - 1e-3)

        # Convert back to Cartesian, maintaining SAME radius (zoom)
        vx = r * jnp.sin(phi) * jnp.cos(theta)
        vy = r * jnp.sin(phi) * jnp.sin(theta)
        vz = r * jnp.cos(phi)

        # Update camera
        cam.position = f + jnp.array([vx, vy, vz])
        cam.up = up  # preserve the original up vector

    # Set up the plotter for 1st elevation parameter and savefig
    pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
    pl.add_mesh(field_mesh, cmap=cmap,show_scalar_bar=False)
    pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color=coil_color, line_width=5)
    camera_rotation(pl,azim_default,elev_angle=elevParams[0])
    pl.render()
    #pl.show(screenshot='coils_nearaxis_paraview_style_view1.png')
    pl.screenshot(f'{filename}_view1.png')


    # Set up the plotter for 2nd elevation parameter and savefig
    pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
    pl.add_mesh(field_mesh, cmap=cmap,show_scalar_bar=False)
    pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color=coil_color, line_width=5)
    camera_rotation(pl,azim_default,elev_angle=elevParams[1])
    pl.render()
    #pl.show(screenshot='coils_nearaxis_paraview_style_view2.png')
    pl.screenshot(f'{filename}_view2.png')

    # Set up the plotter for 2nd elevation parameter and savefig
    pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
    pl.add_mesh(field_mesh, cmap=cmap,show_scalar_bar=False)
    pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color=coil_color, line_width=5)
    camera_rotation(pl,azim_default,elevParams[2])
    pl.render()
    #pl.show(screenshot='coils_nearaxis_paraview_style_view3.png')
    pl.screenshot(f'{filename}_view3.png')


df = pd.read_csv("../data/XGStels/XGStels_balanced.csv")

half_total_key = {
    3 : 1060,
    4 : 725,
    2 : 400,
    5 : 275,
    1 : 40 
}


folder = "data/stel_coils_images"


curr_key = defaultdict(int)
other_half = defaultdict(int)

for idx, row in enumerate(df.itertuples(), start=1):
    if idx == 1:
        continue
    print(idx)

    nfp = int(row.nfp)
    if curr_key[nfp] >= half_total_key[nfp]:
        other_half[nfp] += 1
        cmap = "grays"
        coil_color = "grey"
    else:
        curr_key[nfp] += 1
        cmap = random.choice(["jet", "viridis"])
        coil_color = "gold"
    
    field_nearaxis = create_stel(row.rc1, row.rc2, row.rc3, row.zs1, row.zs2, row.zs3, row.etabar, nfp)
    print("Created nearaxis field")

    filename = f"image_b0_{idx}_{nfp}" if cmap != "grays" else f"image_b1_{idx}_{nfp}"
    generate_stel_image(nfp, cmap, coil_color, field_nearaxis, filename)
    break
