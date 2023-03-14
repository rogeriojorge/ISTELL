#!/usr/bin/env python3
import os
import glob
import shutil
import numpy as np
from pathlib import Path
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec,  QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import CurveSurfaceDistance, curves_to_vtk, create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries
mpi = MpiPartition()
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print
######## INPUT PARAMETERS ########
ncoils=7
CS_THRESHOLD = 0.003
CS_WEIGHT = 1e30
max_nfev = 20
iota_target = 0.187
iota_weight = 5e1
aspect_target = 7.0
aspect_weight = 5e-4
max_modes = [3]
rel_step = 1e-5
abs_step = 1e-7
ISTTOK_R0 = 0.46
ISTTOK_R1 = 0.085
ntheta_VMEC = 51
nphi_VMEC = 51
numquadpoints = 91
ftol=1e-4
diff_method = 'centered'
######## END INPUT PARAMETERS ########
### Go to results folder
results_path = os.path.join(os.path.dirname(__file__), 'results')
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)
### Get VMEC surface
filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_initial')
vmec = Vmec(filename, mpi=mpi, verbose=False, ntheta=ntheta_VMEC, nphi=nphi_VMEC, range_surface='half period')
s = vmec.boundary
### Create coils
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=ISTTOK_R0, R1=ISTTOK_R1, order=1, numquadpoints=numquadpoints)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()
# coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
# curves = [c.curve for c in coils]
# curves_to_vtk(curves, 'curves_init')
curves_to_vtk(base_curves, 'curves_init')
s.to_vtk("surf_init")
### Optimize
surf = vmec.boundary
Jcsdist = CurveSurfaceDistance(base_curves, surf, CS_THRESHOLD)
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
for max_mode in max_modes:
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Major radius
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, aspect_weight),
                                            (qs.residuals, 0, 5e0),
                                            (vmec.mean_iota, iota_target, iota_weight),
                                            (Jcsdist.J, 0, CS_WEIGHT)])
    pprint("Iota before optimization:", vmec.mean_iota())
    pprint("Distance to surfaces before optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J before optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective before optimization:", qs.total())
    pprint("Aspect ratio before optimization:", vmec.aspect())
    pprint("Total objective before optimization:", prob.objective())
    mpi.comm_world.barrier()
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step, abs_step=abs_step, max_nfev=max_nfev, diff_method=diff_method, ftol=ftol)
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final iota:", vmec.mean_iota())
    pprint("Distance to surfaces after optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J after optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective after optimization:", qs.total())
    pprint("Aspect ratio after optimization:", vmec.aspect())
    pprint("Total objective after optimization:", prob.objective())
    s.to_vtk(f"surf_maxmode{max_mode}")
    # vmec.indata.ns_array[:3]    = [  16,    51,    101]
    # vmec.indata.niter_array[:3] = [ 2000,  3000, 20000]
    # vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
    vmec.write_input(f'input.ISTTOK_maxmode{max_mode}')
### Write result
vmec.indata.ns_array[:3]    = [  16,    51,    101]
vmec.indata.niter_array[:3] = [ 2000,  3000, 20000]
vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
vmec.write_input(f'input.ISTTOK_final')
vmec = Vmec('input.ISTTOK_final', mpi=mpi, verbose=True, ntheta=ntheta_VMEC, nphi=nphi_VMEC, range_surface='half period')
vmec.run()
s = vmec.boundary
s.to_vtk("surf_final")
if mpi.proc0_world:
    shutil.move(f"wout_ISTTOK_final_000_000000.nc", f"wout_ISTTOK_final.nc")
    os.remove(f'input.ISTTOK_final_000_000000')
# Remove spurious files
if mpi.proc0_world:
    for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"objective_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"residuals_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
    for objective_file in glob.glob(f"threed*"): os.remove(objective_file)