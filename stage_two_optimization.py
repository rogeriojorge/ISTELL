#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

QA_or_QH = 'QA'

R0 = 1.0
R1 = 0.5 if QA_or_QH == 'QA' else 0.5
order = 8 if QA_or_QH == 'QA' else 10
ncoils = 4 if QA_or_QH == 'QA' else 4
do_long_opt = False
extend_distance = 0.12
use_nfp3 = True
opt_method = 'Powell' if QA_or_QH == 'QA' else 'Powell'

LENGTH_WEIGHT = Weight(1e-1) if QA_or_QH == 'QA' else Weight(1e-1)
LENGTH_THRESHOLD = 12 if QA_or_QH == 'QA' else 13
CS_THRESHOLD = 0.02
CS_WEIGHT = 1e17
CC_THRESHOLD = 0.1 if QA_or_QH == 'QA' else 0.1
CC_WEIGHT = 1e4
CURVATURE_THRESHOLD = 15 if QA_or_QH == 'QA' else 20
CURVATURE_WEIGHT = Weight(1e-6)
MSC_THRESHOLD = 15 if QA_or_QH == 'QA' else 20
MSC_WEIGHT = Weight(1e-6)
MAXITER = 1e2
MAXFEV = 3e3

results_path = os.path.join(os.path.dirname(__file__), 'results_'+QA_or_QH+'_nfp3' if use_nfp3 else '')
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)
filename = 'input.ISTTOK_final_' + QA_or_QH
# Initialize the boundary magnetic surface:
nphi = 26
ntheta = 26

s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
s_full = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi*2*s.nfp, ntheta=ntheta)

# # Create a circular vessel:
# with open('../input.vessel', 'r') as file: content = file.read()
# content = content.replace(f'NFP = 4', f'NFP = {s.nfp}')
# with open('../input.vessel', 'w') as file: file.write(content)
# s_vessel = SurfaceRZFourier.from_vmec_input('../input.vessel', range="half period", nphi=nphi, ntheta=ntheta)
# s_vessel_full = SurfaceRZFourier.from_vmec_input('../input.vessel', range="full torus", nphi=nphi*3, ntheta=ntheta)
# content = content.replace(f'NFP = {s.nfp}', f'NFP = 4')
# with open('../input.vessel', 'w') as file: file.write(content)

# Create a vessel from QH input:
s_vessel1 = SurfaceRZFourier.from_vmec_input(f'../results_QH{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QH', range="half period", nphi=nphi, ntheta=ntheta)
s_vessel1.extend_via_normal(extend_distance)
s_vessel_full1 = SurfaceRZFourier.from_vmec_input(f'../results_QH{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QH', range="full torus", nphi=nphi*2*3, ntheta=ntheta)
s_vessel_full1.extend_via_normal(extend_distance)
s_vessel2 = SurfaceRZFourier.from_vmec_input(f'../results_QA{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QA', range="half period", nphi=nphi, ntheta=ntheta)
s_vessel2.extend_via_normal(extend_distance)
s_vessel_full2 = SurfaceRZFourier.from_vmec_input(f'../results_QA{"_nfp3" if use_nfp3 else ""}/input.ISTTOK_final_QA', range="full torus", nphi=nphi*2*3, ntheta=ntheta)
s_vessel_full2.extend_via_normal(extend_distance)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, "curves_init_"+QA_or_QH)
pointData = {"B_N": np.sum(bs.B().reshape((nphi*2*s.nfp, ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk("surf_init_"+QA_or_QH, extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("surf_init_nfp_"+QA_or_QH, extra_data=pointData)
# s_vessel_full.to_vtk("surf_vessel_"+QA_or_QH)
# s_vessel.to_vtk("surf_vessel_nfp_"+QA_or_QH)
s_vessel_full1.to_vtk("surf_vessel1_"+QA_or_QH)
s_vessel1.to_vtk("surf_vessel1_nfp_"+QA_or_QH)
s_vessel_full2.to_vtk("surf_vessel2_"+QA_or_QH)
s_vessel2.to_vtk("surf_vessel2_nfp_"+QA_or_QH)
curves_to_vtk(base_curves, "curves_init_nfp_"+QA_or_QH)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
# Jcsdist  = CurveSurfaceDistance(curves, s_vessel_full, CS_THRESHOLD)
Jcsdist1 = CurveSurfaceDistance(base_curves, s_vessel1, CS_THRESHOLD)
Jcsdist2 = CurveSurfaceDistance(base_curves, s_vessel2, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + CS_WEIGHT * Jcsdist1 \
    + CS_WEIGHT * Jcsdist2 \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max")

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    outstr = f"J={J:.1e}, Jf={jf:.1e}"
    # BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    # outstr = f", ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len="
    outstr += f"sum([{cl_string}])"
    outstr += f"={sum(J.J() for J in Jls):.1f}"
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    # outstr += f", C-S-Sep1={Jcsdist1.shortest_distance():.2f}"
    # outstr += f", C-S-Sep2={Jcsdist2.shortest_distance():.2f}"
    # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

f = fun
dofs = JF.x
res = minimize(fun, dofs, jac=True, method=opt_method, options={'maxiter': MAXITER, 'maxfev': MAXFEV}, tol=1e-15)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, "curves_opt_short_"+QA_or_QH)
curves_to_vtk(base_curves, "curves_opt_short_nfp_"+QA_or_QH)
pointData = {"B_N": np.sum(bs.B().reshape((nphi*2*s.nfp, ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk("surf_opt_short_"+QA_or_QH, extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk("surf_opt_short_nfp_"+QA_or_QH, extra_data=pointData)

if do_long_opt:
    bs.set_points(s.gamma().reshape((-1, 3)))
    dofs = res.x
    LENGTH_WEIGHT *= 0.1
    CURVATURE_WEIGHT *= 0.1
    MSC_WEIGHT *= 0.1
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER, 'maxfev': MAXFEV}, tol=1e-15)

    bs.set_points(s_full.gamma().reshape((-1, 3)))
    curves_to_vtk(curves, "curves_opt_long_"+QA_or_QH)
    curves_to_vtk(base_curves, "curves_opt_long_nfp_"+QA_or_QH)
    pointData = {"B_N": np.sum(bs.B().reshape((nphi*2*s.nfp, ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
    s_full.to_vtk("Surf_opt_long_"+QA_or_QH, extra_data=pointData)

    bs.set_points(s.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk("surf_opt_long_nfp_"+QA_or_QH, extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(f"biot_savart_opt_{QA_or_QH}.json")
