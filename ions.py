import numpy as np
from scipy.special import erfc
from ase.geometry import cellpar_to_cell
from ase.neighborlist import neighbor_list
from ase.units import Bohr

def ions_efg(atoms, alpha=1.6733200530681511, gmax=20, rmax=20):
    """
    Compute the electric field gradient (EFG) tensor at each atomic site
    using Ewald summation.

    Parameters
    ----------
    atoms : ASE Atoms
        Contains unit-cell, periodic boundary conditions (must be PBC),
        atomic positions, and charges in atoms.get_initial_charges().
    alpha : float or None
        Ewald parameter. If None, chosen automatically.
    gmax : int
        Maximum reciprocal lattice index.
    rmax : int
        Maximum real-space lattice index.

    Returns
    -------
    efg : (N, 3, 3) array
        EFG tensor for each atom.
    """

    cell = atoms.cell / Bohr
    rcell = 2 * np.pi * atoms.cell.reciprocal() * Bohr
    vol = np.linalg.det(cell)
    positions = atoms.get_positions() / Bohr
    charges = atoms.get_atomic_numbers()
    nat = len(atoms)

    if alpha is None:
        # Empirical good default
        alpha = (np.pi * nat / vol)**(1/3)
        print('Alpha: ', alpha)

    efg = np.zeros((nat, 3, 3))

    # ============================================================
    # REAL-SPACE contribution
    # ============================================================

    # I-atoms, J-atoms, N-orms, (vector) D-istances
    I, J, N, D = neighbor_list('ijdD',atoms,rmax)
    for i, j, n, d in zip(I,J,N,D):
        Xab = 3.0 * np.outer(d,d) - (n**2)  * np.eye(3)

        term1 = Xab * (erfc(alpha*n) / n**5)
        term2 = (2*alpha/np.sqrt(np.pi)) * Xab * (np.exp(-(alpha*n)**2) / n**4)

        efg[i] += charges[j] * (term1 + term2)
        
    # print('Real part:', efg)

    # ============================================================
    # RECIPROCAL-SPACE contribution
    # ============================================================
    X, Y, Z = np.ogrid[-gmax:gmax+1, -gmax:gmax+1, -gmax:gmax+1]
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    mask = R <= gmax
    
    # To get actual coordinate lists:
    xs, ys, zs = np.where(mask)
    coords = np.column_stack((xs - gmax, ys - gmax, zs - gmax))

    # G vectors
    g = coords @ rcell
    # Norm^2 of G vectors
    gg = np.einsum('ij,ij->i',g, g)
    # Outer product of G vectors
    ogg = np.einsum('ni,nj->nij',g,g)

    # Structure factor
    strfact = np.sum(charges[:,None]  * np.exp(-1.j * positions.dot(g.T)), axis=0)

    # Ewald factor
    factor = (4*np.pi / vol) * np.exp(-gg/(4.*alpha**2))
    
    for at in range(nat):
    
        # Atom at
        phase = np.einsum('ij,j->i',g, positions[at])
            
        # UNDERSTANDABLE
        #efg = np.zeros([3,3],dtype=complex)
        #for i in range(515):
        #    if gg[i] < 1e-8:
        #        continue
        #    efg += -(ogg[i] * a[i] / gg[i]) + np.eye(3)*a[i]/3.
    
        # FASTer
        # remove g=0
        mask = gg >= 1e-8
        a = factor * strfact * np.exp(-1.j*phase) # (np.cos(phase) - 1j*np.sin(phase))
        coef = a[mask] / gg[mask]

        # Sum of -(ogg[i] * a[i] / gg[i])
        term1 = -np.einsum("i,ijk->jk", coef, ogg[mask])
        
        # Sum of np.eye(3) * a[i] / 3
        term2 = np.eye(3) * (a[mask].sum() / 3.)
        
        efg[at] += np.real_if_close(term1 + term2, tol=1e-8)

    return efg

#if __name__ == '__main__':
#    from ase.build import bulk
#    from ase.io import read
#    from ase import Atoms
#    
#    # NaCl example (charges required)
#    atoms = bulk("NaCl", "rocksalt", a=5.64)
#    charges = [1 if s == "Na" else -1 for s in atoms.get_chemical_symbols()]
#    atoms.set_initial_charges(charges)
#    
#    
#    efg = ions_efg(atoms, gmax=30, rmax=10)
#    
#    print("EFG at each site:")
#    for i, E in enumerate(efg):
#        print(i, "\n", E)
    
