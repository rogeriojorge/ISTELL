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
N_interpolation = 30
N_extra = 2
x_centered = 5
def interp_function(x): return 1/(1+np.exp(-x+x_centered))
interp_factors = sorted(np.concatenate([np.linspace(0,0.03,N_extra,endpoint=True),interp_function(np.linspace(0, x_centered*2, N_interpolation, endpoint=True)),np.linspace(0.97,1.0,N_extra,endpoint=True)]))
# plt.plot(interp_factors)
# plt.show()
# exit()
ncoils = 5
output_path = 'surfs_between_nfp3'
this_path = '/Users/rogeriojorge/local/ISTELL'
# results_path = os.path.join(this_path, 'results_single_stage')
results_path = os.path.join(this_path, 'results_both_nfp3')
os.chdir(results_path)

# coils_path = os.path.join(results_path, f'optimal_coils/ncoils_5_order_8_R1_0.36_length_target_3.3_weight_0.0084_max_curvature_1.6e+01_weight_1.7e-06_msc_1.3e+01_weight_0.00013_cc_0.13_weight_1.5e+03')

coils_path = os.path.join(this_path, 'results_single_stage', f'optimal_coils/ncoils_5_order_6_R1_0.49_length_target_3.6_weight_0.0072_max_curvature_2.4e+01_weight_0.0099_msc_9.0_weight_4.6e-06_cc_0.1_weight_3.4e+02')

# filename1 = os.path.join('..', 'results_QH_nfp3', 'input.ISTTOK_final_QH')
# filename2 = os.path.join('..', 'results_QA_nfp3', 'input.ISTTOK_final_QA')
filename1 = os.path.join(this_path, 'results_single_stage', 'input.ISTTOK_final_QH')
filename2 = os.path.join(this_path, 'results_single_stage', 'input.ISTTOK_final_QA')

# Load surfaces
surf1 = SurfaceRZFourier.from_vmec_input(filename1, range="full torus", nphi=nphi, ntheta=ntheta)
surf2 = SurfaceRZFourier.from_vmec_input(filename2, range="full torus", nphi=nphi, ntheta=ntheta)
surf_between = SurfaceRZFourier.from_vmec_input(filename1, range="full torus", nphi=nphi, ntheta=ntheta)
surf1_dofs = surf1.x
surf2_dofs = surf2.x

# Load coils
bs1 = load(os.path.join(coils_path, "biot_savart1.json"))
bs2 = load(os.path.join(coils_path, "biot_savart2.json"))
bs1_dofs = bs1.x
bs2_dofs = bs2.x
currents_1 = np.array([bs1.coils[i].current.get_value() for i in range(ncoils)])
currents_2 = np.array([bs2.coils[i].current.get_value() for i in range(ncoils)])
max_current = max(max(currents_1), max(currents_2))
min_current = min(min(currents_1), min(currents_2))/max_current
print(f"Coil currents 1: {currents_1}")
print(f"Coil currents 2: {currents_2}")

# Read coils mesh
coils_vtu = pv.read(os.path.join(coils_path, "curves_opt_halfnfp.vtu"))

# Create output directory if not exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# Create a list to store the file names of the PNG images
image_files = []

# Convert VTK files to PNG images and store file names
clim=0
for i in range(2 * len(interp_factors) - 1):
    j = 2 * len(interp_factors) - i - 2 if i > len(interp_factors) - 1 else i
    factor = interp_factors[j]
    # print(factor)

    # Interpolate surfaces and currents
    surf_between.x = (1 - factor) * surf1_dofs + factor * surf2_dofs
    bs1.x = (1 - factor) * bs1_dofs + factor * bs2_dofs
    bs1.set_points(surf_between.gamma().reshape((-1, 3)))

    # Calculate surface normals
    BdotN1 = (np.sum(bs1.B().reshape((nphi, ntheta, 3)) * surf_between.unitnormal(), axis=2)) / np.linalg.norm(bs1.B().reshape((nphi, ntheta, 3)), axis=2)
    print(f'max BdotN: {np.max(np.abs(BdotN1))}')
    if clim==0: clim=np.max(np.abs(BdotN1))
    surf_between.to_vtk(os.path.join(output_path, f"surf_between_halfnfp_{i}"), extra_data={"B.n/B": BdotN1[:, :, None]})

    # Plot surfaces and coils
    vtk_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.vts")
    png_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.png")
    surf_between_vtk = pv.read(vtk_file)

    plotter = pv.Plotter(off_screen=True)

    args_cbar = dict(height=0.1, vertical=False, position_x=0.29, position_y=0.03, color="k", title_font_size=24, label_font_size=16)

    surf_mesh = plotter.add_mesh(surf_between_vtk, scalars="B.n/B", cmap="coolwarm", clim=[-clim, clim], scalar_bar_args=args_cbar)
    # Normalize current values
    current_values = np.array([coil.current.get_value() for coil in bs1.coils])
    for coil_index, coil in enumerate(bs1.coils):
        # Extract the points for the current coil
        coil_points = coils_vtu.extract_cells(coil_index)
        
        # Check if the coil has points before plotting
        if coil_points.n_points > 0:
            args_cbar = dict(width=0.05, vertical=True, position_x=0.05, position_y=0.03, color="k", title_font_size=24, label_font_size=16, title='Current')
            plotter.add_mesh(coil_points, line_width=6, label=f"Coil {coil_index}", scalars=[current_values[coil_index]/max_current]*coil_points.n_points, cmap="coolwarm", scalar_bar_args=args_cbar, clim=[min_current, 1])
    # # Add scalar bar for the current values legend
    # plotter.add_scalar_bar(title="Currents", vertical=True, width=0.05, position_x=0.05, position_y=0.1, title_font_size=24, label_font_size=16)

    # Set background to white
    plotter.set_background("white")
    
    # Adjust camera position (rotate and zoom)
    plotter.camera_position = (-5.1,-0.9,2)#[(7, -2, 0), (0, 0, 0), (0, 0, 1)]  # Example camera position (adjust as needed)
    # plotter.camera_clipping_range = [5, 20]  # Adjust the clipping range to zoom in
    plotter.camera.zoom(1.6)

    plotter.show(screenshot=png_file)
    image_files.append(png_file)
    # exit()
    
# Create a gif from the PNG images
gif_file = os.path.join(output_path, "surf_between_animation.gif")
with imageio.get_writer(gif_file, mode='I') as writer:
    for image_file in image_files:
        image = imageio.v2.imread(image_file)
        writer.append_data(image)

# Print the path to the generated gif
print(f"GIF created: {gif_file}")

# Remove the VTS and PNG files
for vtk_file in os.listdir(output_path):
    if vtk_file.endswith(".vts") or vtk_file.endswith(".png"):
        os.remove(os.path.join(output_path, vtk_file))