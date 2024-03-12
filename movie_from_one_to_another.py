#!/usr/bin/env python3
import os
import imageio
import numpy as np
import pyvista as pv
from simsopt import load
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from simsopt.geo import SurfaceRZFourier

# Constants
nphi = 60
ntheta = 60
N_interpolation = 10
ncoils = 4
output_path = 'surfs_between_nfp3'
this_path = '/Users/rogeriojorge/local/ISTELL'
results_path = os.path.join(this_path, 'results_both_nfp3')
os.chdir(results_path)
coils_path = os.path.join(results_path, f'optimal_coils/ncoils_{ncoils}_order_7_R1_0.41_length_target_3.5_weight_0.0014_max_curvature_9.4_weight_0.00077_msc_2.5e+01_weight_0.00018_cc_0.1_weight_1.2e+01')
filename1 = os.path.join('..', 'results_QH_nfp3', 'input.ISTTOK_final_QH')
filename2 = os.path.join('..', 'results_QA_nfp3', 'input.ISTTOK_final_QA')

# Load surfaces
surf1 = SurfaceRZFourier.from_vmec_input(filename1, range="half period", nphi=nphi, ntheta=ntheta)
surf2 = SurfaceRZFourier.from_vmec_input(filename2, range="half period", nphi=nphi, ntheta=ntheta)
surf_between = SurfaceRZFourier.from_vmec_input(filename1, range="half period", nphi=nphi, ntheta=ntheta)
surf1_dofs = surf1.x
surf2_dofs = surf2.x

# Load coils
bs1 = load(os.path.join(coils_path, "biot_savart1.json"))
bs2 = load(os.path.join(coils_path, "biot_savart2.json"))
bs1_dofs = bs1.x
bs2_dofs = bs2.x
print(f"Coil currents 1: {[bs2.coils[i].current.get_value() for i in range(ncoils)]}")
print(f"Coil currents 2: {[bs1.coils[i].current.get_value() for i in range(ncoils)]}")

# Read coils mesh
coils_vtu = pv.read(os.path.join(coils_path, "curves_opt_halfnfp.vtu"))

# Create output directory if not exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# Create a list to store the file names of the PNG images
image_files = []

# Convert VTK files to PNG images and store file names
interp_factors = np.linspace(0, 1, N_interpolation, endpoint=True)
for i in range(2 * N_interpolation - 1):
    j = 2 * N_interpolation - i - 2 if i > N_interpolation - 1 else i
    factor = interp_factors[j]

    # Interpolate surfaces and currents
    surf_between.x = (1 - factor) * surf1_dofs + factor * surf2_dofs
    bs1.x = (1 - factor) * bs1_dofs + factor * bs2_dofs
    bs1.set_points(surf_between.gamma().reshape((-1, 3)))

    # Calculate surface normals
    BdotN1 = (np.sum(bs1.B().reshape((nphi, ntheta, 3)) * surf_between.unitnormal(), axis=2)) / np.linalg.norm(bs1.B().reshape((nphi, ntheta, 3)), axis=2)
    surf_between.to_vtk(os.path.join(output_path, f"surf_between_halfnfp_{i}"), extra_data={"B.n/B": BdotN1[:, :, None]})

    # Plot surfaces and coils
    vtk_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.vts")
    png_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.png")
    surf_between_vtk = pv.read(vtk_file)

    plotter = pv.Plotter(off_screen=True)
    surf_mesh = plotter.add_mesh(surf_between_vtk, scalars="B.n/B", cmap="coolwarm")

    # Create a colormap
    cmap = plt.cm.coolwarm

    # Normalize current values
    current_values = np.array([coil.current.get_value() for coil in bs1.coils])
    normalized_currents = ((current_values - np.min(current_values)) / (np.max(current_values) - np.min(current_values)))

    # Map normalized currents to colors
    colors = [mcolors.rgb2hex(cmap(norm)) for norm in normalized_currents]

    for coil_index, coil in enumerate(bs1.coils):
        # Extract the points for the current coil
        coil_points = coils_vtu.extract_cells(coil_index)
        
        # Check if the coil has points before plotting
        if coil_points.n_points > 0:
            # Get the color for the current coil based on its current value
            color = colors[coil_index]

            # Plot the current coil
            plotter.add_mesh(coil_points, color=color, line_width=3)

    # Set background to white
    plotter.set_background("white")
    
    # Adjust camera position (rotate and zoom)
    plotter.camera_position = [(0, 1, 1), (0, 0, 0), (0, 0, 1)]  # Example camera position (adjust as needed)

    plotter.show(screenshot=png_file)
    image_files.append(png_file)
    exit()
    
# Create a gif from the PNG images
gif_file = os.path.join(output_path, "surf_between_animation.gif")
with imageio.get_writer(gif_file, mode='I') as writer:
    for image_file in image_files:
        image = imageio.v2.imread(image_file)
        writer.append_data(image)

# Print the path to the generated gif
print(f"GIF created: {gif_file}")
