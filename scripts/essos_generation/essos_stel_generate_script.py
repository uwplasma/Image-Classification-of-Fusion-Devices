"""
Generate stellarator near-axis images and optimized coil wireframe renderings.
"""

# ==========================
# Imports
# ==========================

import random
from time import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import jax.numpy as jnp
import pyvista as pv

from essos.fields import near_axis
from essos.coils import Coils, Curves
from essos.optimization import optimize_loss_function
from essos.objective_functions import loss_coils_for_nearaxis


# ==========================
# Global Configuration
# ==========================

NPHI_INTERNAL = 51
NTHETA = 41
R_SURFACE = 0.1
R_COILS = 0.4
NCOILS = 2

AZIM_DEFAULT_HELICITY0 = 0
ELEV_ANGLES = [-50, -10, 30]  # view 1, 2, 3 camera elevations


# ==========================
# Utility Functions
# ==========================

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_output_folder() -> Path:
    """Ensure the output image folder exists."""
    folder = get_project_root() / "data" / "stel_coils_images"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ==========================
# Physics / Stellarator Construction
# ==========================

def create_stellarator(rc1, rc2, rc3, zs1, zs2, zs3, etabar, nfp):
    """Construct a near-axis field for a stellarator."""
    rc = [1, rc1, rc2, rc3]
    zs = [0, zs1, zs2, zs3]
    return near_axis(rc=rc, zs=zs, etabar=etabar, nfp=int(nfp), nphi=NPHI_INTERNAL)


def setup_initial_coils(field, nfp):
    """Create initial Fourier-based coil curves around the near-axis field."""
    Raxis, Zaxis, phi = field.R0, field.Z0, field.phi
    Xaxis, Yaxis = Raxis * jnp.cos(phi), Raxis * jnp.sin(phi)

    X1 = R_COILS * jnp.cos(phi)
    Y1 = R_COILS * jnp.sin(phi)

    n_curves = NCOILS
    stellsym = True
    points = jnp.array([Xaxis, Yaxis, Zaxis])

    angles = (jnp.arange(n_curves) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)

    curves = jnp.zeros((n_curves, 3, 11))
    curves = curves.at[:, 0, 0].set(jnp.interp(angles, phi, Xaxis, period=2*jnp.pi/nfp))
    curves = curves.at[:, 0, 2].set(jnp.interp(angles, phi, X1, period=2*jnp.pi/nfp))

    curves = curves.at[:, 1, 0].set(jnp.interp(angles, phi, Yaxis, period=2*jnp.pi/nfp))
    curves = curves.at[:, 1, 2].set(jnp.interp(angles, phi, Y1, period=2*jnp.pi/nfp))

    curves = curves.at[:, 2, 0].set(jnp.interp(angles, phi, Zaxis, period=2*jnp.pi/nfp))
    curves = curves.at[:, 2, 1].set(-R_COILS)

    curves_complete = Curves(curves, n_segments=NTHETA, nfp=nfp, stellsym=stellsym)

    current = 17e5 * field.B0 / nfp / 2
    return Coils(curves=curves_complete, currents=[current] * n_curves)


# ==========================
# Rendering / Camera Helpers
# ==========================

def rotate_camera(plotter, azim=0, elev=0):
    """Rotate PyVista camera with spherical coordinate adjustments."""
    cam = plotter.camera
    pos, f, up = map(jnp.array, (cam.position, cam.focal_point, cam.up))

    v = pos - f
    r = jnp.linalg.norm(v)

    x, y, z = v
    theta = jnp.arctan2(y, x)
    phi = jnp.arccos(z / r)

    theta += jnp.radians(azim)
    phi = jnp.clip(phi + jnp.radians(elev), 1e-3, jnp.pi - 1e-3)

    vx = r * jnp.sin(phi) * jnp.cos(theta)
    vy = r * jnp.sin(phi) * jnp.sin(theta)
    vz = r * jnp.cos(phi)

    cam.position = f + jnp.array([vx, vy, vz])
    cam.up = up


def render_and_save_views(field_mesh, coils_mesh, cmap, coil_color, base_filepath, azim):
    """Render three camera angles of the stellarator and save screenshots."""
    for i, elev in enumerate(ELEV_ANGLES, start=1):
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(field_mesh, cmap=cmap, show_scalar_bar=False)
        pl.add_mesh(
            coils_mesh,
            style="wireframe",
            render_lines_as_tubes=True,
            color=coil_color,
            line_width=5,
        )
        rotate_camera(pl, azim, elev)
        pl.render()

        pl.screenshot(f"{base_filepath}_v{i}.png")
        
        # IMPORTANT: free resources
        pl.clear()
        pl.close()

    pv.close_all()  # extra safety

# ==========================
# Main Processing Loop
# ==========================

def process_row(idx, row, folder, curr_key, half_total_key, other_half):
    """Process one dataset row and generate stellarator images."""
    nfp = int(row.nfp)

    # Balance color vs grayscale output
    if curr_key[nfp] >= half_total_key[nfp]:
        other_half[nfp] += 1
        cmap, coil_color, mode = "grays", "grey", "bw"
    else:
        curr_key[nfp] += 1
        cmap = random.choice(["jet", "viridis"])
        coil_color, mode = "gold", "color"

    # Create physics
    field = create_stellarator(row.rc1, row.rc2, row.rc3,
                               row.zs1, row.zs2, row.zs3,
                               row.etabar, nfp)

    base_filepath = folder / f"stel_idx{idx}_nfp{nfp}_cmap-{mode}"

    # Coil optimization
    initial = setup_initial_coils(field, nfp)
    coils_opt = optimize_loss_function(
        loss_coils_for_nearaxis,
        initial_dofs=initial.x,
        coils=initial,
        tolerance_optimization=1e-8,
        maximum_function_evaluations=10,
        field_nearaxis=field,
        max_coil_length=10,
        max_coil_curvature=2,
    )

    coils_opt.to_vtk("coils_optimized")
    field.to_vtk("nearaxis_field", r=R_SURFACE, ntheta=NTHETA, nphi=101)

    field_mesh = pv.read("nearaxis_field.vts")
    coils_mesh = pv.read("coils_optimized.vtu")

    render_and_save_views(field_mesh, coils_mesh, cmap, coil_color, base_filepath, AZIM_DEFAULT_HELICITY0)
    
    # free VTK/PyVista objects
    field_mesh.clear_data()
    coils_mesh.clear_data()
    del field_mesh, coils_mesh

# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    folder = get_output_folder()
    df = pd.read_csv("../../data/XGStels/XGStels_balanced.csv")

    half_total_key = {3: 1060, 4: 725, 2: 400, 5: 275, 1: 40}
    curr_key = defaultdict(int)
    other_half = defaultdict(int)

    for idx, row in enumerate(df.itertuples(), start=1):
        try:
            process_row(idx, row, folder, curr_key, half_total_key, other_half)
            print(f"[OK] idx={idx} nfp={row.nfp}")
        except Exception as e:
            print(f"[ERROR] idx={idx} nfp={row.nfp}: {e}")
