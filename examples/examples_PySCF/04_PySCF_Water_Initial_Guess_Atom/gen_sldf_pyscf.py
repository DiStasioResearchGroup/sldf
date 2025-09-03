import sys
import os
import numpy as np
import pyscf
from pyscf import dft

import sldf

mol = pyscf.gto.Mole()

#mol.atom = '''
#H    0.00000000    0.91087015    0.53855853
#H    0.00000000   -0.88833620    0.55719192
#O    0.00000000    0.00000000    1.29177249
#'''
mol.atom = '''
H 0.0 0.0 0.0
H 0.0 0.0 0.74
'''
mol.basis   = 'ccpvtz'
mol.verbose = 10
mol.output  = 'pyscf.log'
mol.build()

dm = pyscf.scf.hf.init_guess_by_atom(mol)

# Grid settings
grids = pyscf.dft.gen_grid.Grids(mol)
grids.atom_grid = (99, 590)
grids.prune = None
grids.build()

# Grid information
coords = grids.coords
weights = grids.weights

# AO values
ao = pyscf.dft.numint.eval_ao(mol, coords, deriv=1)

# Electron density and gradients
rho, dx_rho, dy_rho, dz_rho = pyscf.dft.numint.eval_rho(mol, ao, dm, xctype='GGA')

# Evaluate reduced density gradient, s
s = np.sqrt(dx_rho ** 2 + dy_rho ** 2 + dz_rho ** 2) / (rho ** (4/3))

nsp = 20
sldf = sldf.calc_SLDF(rho, s, weights, nsp)

np.savetxt('sldf.csv', sldf)

print('SLDF is')
print(sldf)

Ex_sldf = np.sum(sldf[:nsp])
Ec_sldf = np.sum(sldf[nsp:])

# Sanity check SLDF XC energies
ex, vx, _, _ = pyscf.dft.libxc.eval_xc('lda,', rho)
ec, vc, _, _ = pyscf.dft.libxc.eval_xc(',pw_mod', rho)

Ex = np.einsum('i,i,i->', ex, rho, weights)
Ec = np.einsum('i,i,i->', ec, rho, weights)

if np.allclose(Ex_sldf, Ex) and np.allclose(Ec_sldf, Ec):
    print('SLDF generation is successful!')
    print('SLDF is saved to sldf.csv')
    print(f'Sum of Exchange SLDF = {Ex_sldf:.3f}')
    print(f'Sum of Correlation SLDF = {Ec_sldf:.3f}')
else:
    print('Warning: Sum of SLDF is not consistent with LDA XC energies.')
