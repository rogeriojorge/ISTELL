#!/usr/bin/env python3
import os
import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
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
CS_THRESHOLD = 0.005
CS_WEIGHT = 1e30
max_nfev = 50
iota_target = 0.23
iota_weight = 3e0
aspect_target = 6.5
aspect_weight = 1e-3
max_modes = [1, 2, 3]
rel_step = 1e-3
abs_step = 1e-5
######## END INPUT PARAMETERS ########
filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_initial')
vmec = Vmec(filename, mpi=mpi, verbose=False, ntheta=50, nphi=160)
s = vmec.boundary
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=0.46, R1=0.085, order=1, numquadpoints=100)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
curves = [c.curve for c in coils]
curves_to_vtk(curves, 'curves_init')
s.to_vtk("surf_init")
for max_mode in max_modes:
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Major radius
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    qs = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, aspect_weight),
                                            (qs.residuals, 0, 1),
                                            (vmec.mean_iota, iota_target, iota_weight),
                                            (Jcsdist.J, 0, CS_WEIGHT)])
    pprint("Iota before optimization:", vmec.mean_iota())
    pprint("Distance to surfaces before optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J before optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective before optimization:", qs.total())
    pprint("Aspect ratio before optimization:", vmec.aspect())
    pprint("Total objective before optimization:", prob.objective())
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step, abs_step=abs_step, max_nfev=max_nfev)
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final iota:", vmec.mean_iota())
    pprint("Distance to surfaces after optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J after optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective after optimization:", qs.total())
    pprint("Aspect ratio after optimization:", vmec.aspect())
    pprint("Total objective after optimization:", prob.objective())
    s.to_vtk(f"surf_maxmode{max_mode}")
    vmec.indata.ns_array[:3]    = [  16,    51,    101]
    vmec.indata.niter_array[:3] = [ 2000,  3000, 20000]
    vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
    vmec.write_input(f'input.ISTTOK_maxmode{max_mode}')
vmec.indata.ns_array[:3]    = [  16,    51,    101]
vmec.indata.niter_array[:3] = [ 2000,  3000, 20000]
vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
vmec.write_input(f'input.ISTTOK_final')
vmec = Vmec('input.ISTTOK_final')
vmec.run()
s = vmec.boundary
s.to_vtk("surf_final")