import numpy as np
from ions import ions_efg
from charge import charge_efg
from ase.io.cube import read_cube_data

rho, atoms = read_cube_data('charge.cube')

#atoms.set_scaled_positions(np.array([[0.,0.,0.], [0.5 ,0.5, 0.5]]))

ions = ions_efg(atoms, gmax=60, rmax=20)
electrons = charge_efg(atoms, rho)

print('ions',ions)
print('electrons', electrons)
print(ions+electrons)
