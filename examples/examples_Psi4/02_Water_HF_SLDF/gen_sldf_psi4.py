import sys
import os
import psi4
import numpy as np

import sldf

psi4.set_memory('1000 MB')

mol = psi4.geometry('''
0 1
H    0.00000000    0.91087015    0.53855853
H    0.00000000   -0.88833620    0.55719192
O    0.00000000    0.00000000    1.29177249
symmetry c1
no_reorient
no_com
''')

psi4.set_options(
        {'basis': 'cc-pVTZ',
         'e_convergence': 1e-12,
         'dft_radial_points': 99,
         'dft_spherical_points': 590
         })

psi4.core.set_output_file('psi4.log')
e, wfn = psi4.energy('SCF', return_wfn=True)

D_psi4_mat = wfn.Da()
D = np.array(D_psi4_mat)

# Create a dummy superfunctional object to compute electron density 
# Set its potential based on previously calculated density matrix
# Must be a GGA or above which calculates density gradient
sup = psi4.driver.dft.build_superfunctional('PBE', True)[0]
Vpot = psi4.core.VBase.build(wfn.basisset(), sup, 'RV')
Vpot.initialize()
Vpot.set_D([D_psi4_mat])
Vpot.properties()[0].set_pointers(D_psi4_mat)

points_func = Vpot.properties()[0]

rho = []
dx_rho = []
dy_rho = []
dz_rho = []
weights = []
tau = []

# Reference: psi4numpy/Tutorials/04_Density_Functional_Theory/4c_GGA_and_Meta_GGA.ipynb
# Loop over the blocks
for i in range(Vpot.nblocks()):
    
    # Obtain block information
    block = Vpot.get_block(i)
    points_func.compute_points(block)
    npoints = block.npoints()
    lpos = np.array(block.functions_local_to_global())
    
    # Obtain the grid weight
    w_i = np.array(block.w())

    # Compute phi and derivatives
    phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

    phi_x = np.array(points_func.basis_values()["PHI_X"])[:npoints, :lpos.shape[0]]
    phi_y = np.array(points_func.basis_values()["PHI_Y"])[:npoints, :lpos.shape[0]]
    phi_z = np.array(points_func.basis_values()["PHI_Z"])[:npoints, :lpos.shape[0]]
    
    # Build a local slice of D
    lD = D[(lpos[:, None], lpos)]
    
    # Copmute rho and derivatives
    rho_i = 2.0 * np.einsum('pm,mn,pn->p', phi, lD, phi, optimize=True)

    dx_rho_i = 4.0 * np.einsum('pm,mn,pn->p', phi, lD, phi_x, optimize=True)
    dy_rho_i = 4.0 * np.einsum('pm,mn,pn->p', phi, lD, phi_y, optimize=True)
    dz_rho_i = 4.0 * np.einsum('pm,mn,pn->p', phi, lD, phi_z, optimize=True)

    # Compute Tau
    tau_i = np.einsum('pm, mn, pn->p', phi_x,lD, phi_x, optimize=True)
    tau_i += np.einsum('pm, mn, pn->p', phi_y,lD, phi_y, optimize=True)
    tau_i += np.einsum('pm, mn, pn->p', phi_z,lD, phi_z, optimize=True)

    # Append
    weights.append(w_i)
    rho.append(rho_i) 
    dx_rho.append(dx_rho_i)
    dy_rho.append(dy_rho_i)
    dz_rho.append(dz_rho_i)
    tau.append(tau_i)

# Electron density and gradients
rho = np.concatenate(rho, axis=0)
dx_rho = np.concatenate(dx_rho, axis=0)
dy_rho = np.concatenate(dy_rho, axis=0)
dz_rho = np.concatenate(dz_rho, axis=0)

s = np.sqrt(dx_rho ** 2 + dy_rho ** 2 + dz_rho ** 2) / (rho ** (4/3))

# Grid weights
weights = np.concatenate(weights, axis=0)

# Kinetic energy density, tau;
# int tau dr = KE
tau = np.concatenate(tau, axis=0)

nsp = 20
sldf = sldf.calc_SLDF(rho, s, weights, nsp)

np.savetxt('sldf.csv', sldf)

print('SLDF is')
print(sldf)

Ex_sldf = np.sum(sldf[:nsp])
Ec_sldf = np.sum(sldf[nsp:])

print('SLDF generation is successful!')
print('SLDF is saved to sldf.csv')
print(f'Sum of Exchange SLDF = {Ex_sldf:.3f}')
print(f'Sum of Correlation SLDF = {Ec_sldf:.3f}')
