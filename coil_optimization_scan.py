#!/usr/bin/env python3

# This script runs coil optimizations, one after another, choosing the weights
# and target values from a random distribution. This is effectively a crude form
# of global optimization.

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    SurfaceRZFourier,
    LinkingNumber,
    CurveLength,
    CurveCurveDistance,
    MeanSquaredCurvature,
    LpCurveCurvature,
    CurveSurfaceDistance,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
this_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
parser.add_argument("--ncoils", type=int, default=3)
args = parser.parse_args()

if args.type == 1:
    QA_or_QH = 'QH'
elif args.type == 2:
    QA_or_QH = 'QA'
elif args.type == 3:
    QA_or_QH = 'both'
else:
    raise ValueError('Invalid type')
ncoils = args.ncoils
R1_mean = 0.3
R1_std = 0.3
extend_distance = 0.04
MAXITER = 700
use_nfp3 = True
opt_method = 'BFGS'
min_length_per_coil = 3.2
max_length_per_coil = 4.3
min_curvature = 8
max_curvature = 25
CC_min = 0.07
CC_max = 0.13
order_min = 5
order_max = 15
nphi = 32
ntheta = 32
CS_THRESHOLD = 0.04
CS_WEIGHT = 1e4

results_path = os.path.join(os.path.dirname(__file__), 'results_'+QA_or_QH+'_nfp3' if use_nfp3 else '')
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)
if QA_or_QH == 'both':
    filename1 = os.path.join('..','results_QH_nfp3','input.ISTTOK_final_QH')
    filename2 = os.path.join('..','results_QA_nfp3','input.ISTTOK_final_QA')
    surf1 = SurfaceRZFourier.from_vmec_input(filename1, range="half period", nphi=nphi, ntheta=ntheta)
    surf2 = SurfaceRZFourier.from_vmec_input(filename2, range="half period", nphi=nphi, ntheta=ntheta)
    nfp = surf1.nfp
    R0 = np.mean((surf1.get_rc(0, 0),surf2.get_rc(0, 0)))
else:
    filename = 'input.ISTTOK_final_' + QA_or_QH
    surf = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
    nfp = surf.nfp
    R0 = surf.get_rc(0, 0)

nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)

out_dir = os.path.join(results_path,"scan")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Create a vessel from QH/QA input:
s_vessel1 = SurfaceRZFourier.from_vmec_input(f'../../results_QH{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QH', range="half period", nphi=nphi, ntheta=ntheta)
s_vessel1.extend_via_normal(extend_distance)
s_vessel_full1 = SurfaceRZFourier.from_vmec_input(f'../../results_QH{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QH', range="full torus", nphi=nphi*2*nfp, ntheta=ntheta)
s_vessel_full1.extend_via_normal(extend_distance)
s_vessel2 = SurfaceRZFourier.from_vmec_input(f'../../results_QA{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QA', range="half period", nphi=nphi, ntheta=ntheta)
s_vessel2.extend_via_normal(extend_distance)
s_vessel_full2 = SurfaceRZFourier.from_vmec_input(f'../../results_QA{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QA', range="full torus", nphi=nphi*2*nfp, ntheta=ntheta)
s_vessel_full2.extend_via_normal(extend_distance)

# if surf_vessel is not created yet as a file, create it
if not os.path.isfile(os.path.join(results_path,"surf_vessel1_"+QA_or_QH)):
    s_vessel_full1.to_vtk(os.path.join(results_path,"surf_vessel1_"+QA_or_QH))
    s_vessel1.to_vtk(os.path.join(results_path,"surf_vessel1_nfp_"+QA_or_QH))
    s_vessel_full2.to_vtk(os.path.join(results_path,"surf_vessel2_"+QA_or_QH))
    s_vessel2.to_vtk(os.path.join(results_path,"surf_vessel2_nfp_"+QA_or_QH))

# Create a copy of the surface that is closed in theta and phi, and covers the
# full torus toroidally. This is nice for visualization.
if QA_or_QH == 'both':
    surf_big1 = SurfaceRZFourier(dofs=surf1.dofs,nfp=nfp, mpol=surf1.mpol,ntor=surf1.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)
    surf_big2 = SurfaceRZFourier(dofs=surf2.dofs,nfp=nfp, mpol=surf2.mpol,ntor=surf2.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)
else:
    surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=nfp, mpol=surf.mpol,ntor=surf.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

def run_optimization(
    R1,
    order,
    length_target,
    length_weight,
    max_curvature_threshold,
    max_curvature_weight,
    msc_threshold,
    msc_weight,
    cc_threshold,
    cc_weight,
    index,
):
    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index+1}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    # Directory for output
    new_OUT_DIR = directory + "/"
    os.mkdir(directory)

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=order * 16)
    
    def process_surface_and_flux(bs, surf, surf_big=None, new_OUT_DIR="", prefix=""):
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        maxBdotN = np.max(np.abs(BdotN))
        pointData = {"B.N/B": BdotN[:, :, None]}
        surf.to_vtk(new_OUT_DIR + prefix + "halfnfp", extra_data=pointData)
        if surf_big is not None:
            bs.set_points(surf_big.gamma().reshape((-1, 3)))
            Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
            BdotN = (np.sum(Bbs * surf_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
            pointData = {"B.N/B": BdotN[:, :, None]}
            surf_big.to_vtk(new_OUT_DIR + prefix + "big", extra_data=pointData)
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Jf = SquaredFlux(surf, bs, definition="local")
        return Jf, maxBdotN
    
    if QA_or_QH in ['QH', 'QA']:
        base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
        # base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        curves = [c.curve for c in coils]
        curves_to_vtk(curves, new_OUT_DIR + "curves_init", close=True)
        bs = BiotSavart(coils)
        Jf_total, _ = process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=new_OUT_DIR, prefix='surf_init_')

    elif QA_or_QH == 'both':
        base_currents1 = [Current(1.0) * (1e5) for i in range(ncoils)]
        base_currents2 = [Current(1.0) * (1e5) for i in range(ncoils)]
        coils1 = coils_via_symmetries(base_curves, base_currents1, nfp, True)
        coils2 = coils_via_symmetries(base_curves, base_currents2, nfp, True)
        curves = [c.curve for c in coils1]
        curves_to_vtk(curves, new_OUT_DIR + "curves_init", close=True)
        bs1 = BiotSavart(coils1)
        bs2 = BiotSavart(coils2)
        Jf1, _ = process_surface_and_flux(bs1, surf1, surf_big=surf_big1, new_OUT_DIR=new_OUT_DIR, prefix="surf1_init_")
        Jf2, _ = process_surface_and_flux(bs2, surf2, surf_big=surf_big2, new_OUT_DIR=new_OUT_DIR, prefix="surf2_init_")
        Jf_total = Jf1 + Jf2

    # Define the individual terms objective function:
    
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist1 = CurveSurfaceDistance(curves, s_vessel1, CS_THRESHOLD)
    Jcsdist2 = CurveSurfaceDistance(curves, s_vessel2, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = (
        Jf_total
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        + CS_WEIGHT * Jcsdist1
        + CS_WEIGHT * Jcsdist2
        + max_curvature_weight * sum(Jcs)
        + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )
    
    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf_total.J()
        if QA_or_QH in ['QH', 'QA']:
            Bbs = bs.B().reshape((nphi, ntheta, 3))
            BdotN = np.max(np.abs((np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)))
        elif QA_or_QH == 'both':
            Bbs1 = bs1.B().reshape((nphi, ntheta, 3))
            BdotN1 = np.max(np.abs((np.sum(Bbs1 * surf1.unitnormal(), axis=2)) / np.linalg.norm(Bbs1, axis=2)))
            Bbs2 = bs2.B().reshape((nphi, ntheta, 3))
            BdotN2 = np.max(np.abs((np.sum(Bbs2 * surf2.unitnormal(), axis=2)) / np.linalg.norm(Bbs2, axis=2)))
            BdotN = max(BdotN1, BdotN2)
        outstr = f"{iteration:4} J={J:.1e}, Jf={jf:.1e}"
        outstr += f", max⟨B·n⟩/B={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", L=[{cl_string}], "#={sum(J.J() for J in Jls):.1f}, "
        outstr += f"ϰ=[{kap_string}], msc=[{msc_string}]"
        outstr += f", CC={Jccdist.shortest_distance():.2f}"
        # outstr += f", cs1={Jcsdist1.shortest_distance():.2f}"
        # outstr += f", cs2={Jcsdist2.shortest_distance():.2f}"
        # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    res = minimize( fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 300}, tol=1e-15)
    JF.x = res.x
    print(res.message)
    curves_to_vtk(curves, new_OUT_DIR + "curves_opt", close=True)
    curves_to_vtk(base_curves, new_OUT_DIR + "curves_opt_halfnfp", close=True)

    if QA_or_QH in ['QH', 'QA']:
        Jf_total, BdotN = process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=new_OUT_DIR, prefix='surf_opt_')
        bs.save(new_OUT_DIR + "biot_savart.json")

    elif QA_or_QH == 'both':
        Jf1, maxBdotN1 = process_surface_and_flux(bs1, surf1, surf_big=surf_big1, new_OUT_DIR=new_OUT_DIR, prefix="surf1_opt_")
        Jf2, maxBdotN2 = process_surface_and_flux(bs2, surf2, surf_big=surf_big2, new_OUT_DIR=new_OUT_DIR, prefix="surf2_opt_")
        BdotN = max(maxBdotN1, maxBdotN2)
        Jf_total = Jf1 + Jf2
        bs1.save(new_OUT_DIR + "biot_savart1.json")
        bs2.save(new_OUT_DIR + "biot_savart2.json")

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:

    results = {
        "nfp": nfp,
        "R0": R0,
        "R1": R1,
        "ncoils": ncoils,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf_total.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "max_curvatures": [np.max(c.kappa()) for c in base_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "cc_threshold": cc_threshold,
        "cc_weight": cc_weight,
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents] if QA_or_QH in ['QH', 'QA'] else [c.get_value() for c in base_currents1] + [c.get_value() for c in base_currents2],
        "coil_surface_distance1":  float(Jcsdist1.shortest_distance()),
        "coil_surface_distance2":  float(Jcsdist2.shortest_distance()),
    }

    with open(new_OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


#########################################################################
# Carry out the scan. Below you can adjust the ranges for the random weights and
# thresholds.
#########################################################################


def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min


for index in range(10000):
    # Initial radius of the coils:
    R1 = np.random.rand() * R1_std + R1_mean

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = int(np.round(rand(order_min, order_max)))

    # Target length (per coil!) and weight for the length term in the objective function:
    length_target = rand(min_length_per_coil, max_length_per_coil)
    length_weight = 10.0 ** rand(-3, 1)

    # Threshold and weight for the curvature penalty in the objective function:
    max_curvature_threshold = rand(min_curvature, max_curvature)
    max_curvature_weight = 10.0 ** rand(-6, -2)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    msc_threshold = rand(min_curvature, max_curvature)
    msc_weight = 10.0 ** rand(-6, -3)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    cc_threshold = rand(CC_min, CC_max)
    cc_weight = 10.0 ** rand(-1, 3)

    run_optimization(
        R1,
        order,
        length_target,
        length_weight,
        max_curvature_threshold,
        max_curvature_weight,
        msc_threshold,
        msc_weight,
        cc_threshold,
        cc_weight,
        index,
    )
