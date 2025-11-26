# coding: utf-8
import numpy as np
from ase.io.cube import read_cube_data
from ase.units import Bohr
from scipy.special import erfc

def charge_efg(atoms, rho):
    nx, ny, nz = rho.shape
    rhog = np.fft.fftn(rho, norm='forward') 
    rhog[0,0,0] = 0.
    
    cell = atoms.cell / Bohr
    rcell = 2 * np.pi * atoms.cell.reciprocal() * Bohr
    vol = np.linalg.det(cell)
    positions = atoms.get_positions() / Bohr
    nat = len(atoms)
    
    
    kx = np.fft.fftfreq(nx, 1./nx)
    ky = np.fft.fftfreq(ny, 1./ny)
    kz = np.fft.fftfreq(nz, 1./nz)
    
    KX,KY,KZ = np.meshgrid(kx,ky,kz, indexing='ij')
    Kxyz = np.stack([KX, KY, KZ], axis=0)
    gx, gy, gz = (Kxyz.T @ rcell).T
    gg = gx**2 + gy**2 + gz**2
    gg[0,0,0] = 1e-10
   
    efg = np.zeros((nat, 3, 3)) 
    for i, p in enumerate(positions):
        phase = np.exp(1.j * (p[0]*gx + p[1]*gy + p[2]*gz ))
        dump = erfc(0.01*gg)
    
        rhog_g2_ph_dump = ( rhog/gg ) * phase * dump
    
        vxx = np.sum((gx * gx - (1./3.)*gg ) * rhog_g2_ph_dump )
        vyy = np.sum((gy * gy - (1./3.)*gg ) * rhog_g2_ph_dump )
        vzz = np.sum((gz * gz - (1./3.)*gg ) * rhog_g2_ph_dump )
        vyx = vxy = np.sum((gx * gy ) * rhog_g2_ph_dump )
        vzx = vxz = np.sum((gx * gz ) * rhog_g2_ph_dump )
        vzy = vyz = np.sum((gy * gz ) * rhog_g2_ph_dump )
        
        efg[i] = np.real_if_close(-np.array([[vxx, vxy, vxz], [vyx, vyy, vyz], [vzx, vzy, vzz]]), tol=1e-8)
        
    return efg
