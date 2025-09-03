"""
Core SLDF calculation functions.
"""
import numpy as np
from scipy.interpolate import BSpline 

def calc_SLDF(rho, s, weights, nsp):
    """
    Calculate Semi-Local Density Fingerprints (SLDF) feature vectors.
    Parameters
    ----------
    rho : np.ndarray
        Electron density values at grid points
    s : np.ndarray
        Reduced density gradient values: s = |nabla_rho|/(2(3pi)^(1/3)rho^(4/3))
    weights : np.ndarray
        Grid weights for numerical quadrature
    nsp : int
        Number of splines per component
        
    Returns
    -------
    np.ndarray
        SLDF feature vector:
        - [0:nsp]: Exchange
        - [nsp:2*nsp]: Same-spin correlation
        - [2*nsp:3*nsp]: Opposite-spin correlation

    Notes  
    -----
    Currently, this calculation only supports closed-shell systems 
    with equal alpha and beta densities.
    """

    # Convert to numpy arrays if needed
    try:
        rho = np.asarray(rho, dtype=float)
        s = np.asarray(s, dtype=float)
        weights = np.asarray(weights, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Input arrays must be convertible to float arrays: {e}")
    
    # Flatten arrays
    rho = rho.flatten()
    s = s.flatten()
    weights = weights.flatten()

    # Check array lengths
    if len(rho) != len(s) or len(rho) != len(weights):
        raise ValueError(f"Input arrays must have the same length. Got rho: {len(rho)}, s: {len(s)}, weights: {len(weights)}")

    # Drop electron density when lower than threshold
    i_include = np.where(rho > 1e-16)[0]
    
    if len(i_include) == 0:
        raise ValueError("All density values are below threshold (1e-16). Cannot compute SLDF.")
    
    rho = rho[i_include]
    weights = weights[i_include]
    s = s[i_include]

    # For closed shell systems
    rho_a = rho / 2.0
    rho_b = rho_a

    # Calculate LSDA exchange 
    x_a = calc_LSDA_x(rho_a)
    
    # Calculate LSDA correlation, B97 definition
    c_a = calc_LSDA_c(rho_a, 0)
    c_ab = calc_LSDA_c(rho_a, rho_b) - calc_LSDA_c(rho_a, 0) - calc_LSDA_c(rho_b, 0) 

    # Calculate s and u
    s_a = 2 ** (1/3) * s
    
    ux_a = 0.004 * s_a ** 2 / ( 1.0 + 0.004 * s_a ** 2)
    uc_a = 0.2 * s_a ** 2 / ( 1.0 + 0.2 * s_a ** 2)
    uc_ab = 0.006 * s_a ** 2 / (1.0 + 0.006 * s_a ** 2)

    # For closed-shell system, splines_xa = splines_xb, splines_ca = splines_cb
    splines_xa  = spline_1d(func=x_a,  bvar=ux_a,  weights=weights, nsp=nsp) 
    splines_x   = splines_xa * 2
    splines_ca  = spline_1d(func=c_a,  bvar=uc_a,  weights=weights, nsp=nsp) 
    splines_css = splines_ca * 2
    splines_cos = spline_1d(func=c_ab, bvar=uc_ab, weights=weights, nsp=nsp)
    
    sldf = np.concatenate([splines_x, splines_css, splines_cos])

    return sldf

def calc_LSDA_x(rho_sig):
    """
    Calculate LSDA exchange energy density.
    
    Parameters
    ----------
    rho_sig : np.ndarray
        Electron density for spin density rho_sigma
        
    Returns
    -------
    np.ndarray
        Exchange energy density 
        
    """
    x_sig = - 1.5 * (3 / (4*np.pi)) ** (1/3) * rho_sig ** (4/3) 
    return x_sig

def calc_LSDA_c(rho_a, rho_b):
    """
    Calculate LSDA correlation energy density.
    
    Parameters
    ----------
    rho_a : np.ndarray
        Alpha electron density
    rho_b : np.ndarray  
        Beta electron density
        
    Returns
    -------
    np.ndarray
        Correlation energy density
        
    Notes
    -----
    Uses Perdew-Wang 1992 parameterization with extra precision digits
    
    """
    # Perdew and Wang, 1992
    # Parameters, extra digits added (in Libxc, named as PW_MOD)
    A_0 = 0.0310907
    A_1 = 0.01554535
    A_2 = 0.0168869
    alpha1_0 = 0.21370
    alpha1_1 = 0.20548
    alpha1_2 = 0.11125
    beta1_0 = 7.5957
    beta1_1 = 14.1189
    beta1_2 = 10.357
    beta2_0 = 3.5876
    beta2_1 = 6.1977
    beta2_2 = 3.6231
    beta3_0 = 1.6382
    beta3_1 = 3.3662
    beta3_2 = 0.88026
    beta4_0 = 0.49294
    beta4_1 = 0.62517
    beta4_2 = 0.49671
    fzeta20 = 1.709920934161365617563962776245

    r_s = (3 / (4 * np.pi * (rho_a + rho_b))) ** (1/3)

    # In closed-shell system, zeta is either 0 or 1
    zeta = (rho_a - rho_b) / (rho_a + rho_b)

    def G(r_s, A, alpha1, beta1, beta2, beta3, beta4):
        G = - 2 * A * (1 + alpha1 * r_s) * np.log( 1 + 1 / (2 * A * (beta1 * r_s ** 0.5 + beta2 * r_s + beta3 * r_s ** 1.5 + beta4 * r_s ** 2.0)))
        return G
    
    fzeta = ((1 + zeta) ** (4/3) + (1 - zeta) ** (4/3) - 2) / (2 ** (4/3) - 2)

    eps_c_0 = G(r_s, A_0, alpha1_0, beta1_0, beta2_0, beta3_0, beta4_0) 
    eps_c_1 = G(r_s, A_1, alpha1_1, beta1_1, beta2_1, beta3_1, beta4_1) 
    alpha_c = G(r_s, A_2, alpha1_2, beta1_2, beta2_2, beta3_2, beta4_2)

    eps = eps_c_0 + alpha_c * fzeta * (1 - zeta ** 4) / fzeta20 + (eps_c_1 - eps_c_0) * fzeta * zeta ** 4

    c = eps * (rho_a + rho_b)

    return c

def gen_knots(nsp, b_order=3):
    """
    Generate knot sequences for B-spline basis functions.
    
    Parameters
    ----------
    nsp : int
        Number of spline functions to generate
    b_order : int
        B-spline order. Default is 3 for cubic B-splines
        
    Returns
    -------
    list of np.ndarray
        List containing knot vectors for each B-spline basis function.
        Each knot vector has length (b_order + 2).
        
    """
    # Lower and upper bounds of knots
    imin = - b_order / (nsp - 2 * b_order / 3 - 1)
    imax = (nsp - 1 + b_order / 2) / (nsp - 2 * b_order / 3 - 1)
    step = 1 / (nsp - 3)
    knotslist = np.arange(imin, imax, step)

    knots = []
    for istart in range(nsp):
        iend = istart + b_order + 2
        knots.append(knotslist[istart : iend])

    return knots

def spline_1d(func, bvar, weights, nsp, b_order=3):
    """
    Generate 1D SLDF integrals.
    
    Parameters
    ----------
    func : np.ndarray
        Function values to be projected (e.g., exchange or correlation energy density)
    bvar : np.ndarray  
        B-spline variable 
    weights : np.ndarray
        Grid weights for numerical quadrature
    nsp : int
        Number of splines 
    b_order : int
        B-spline order. Default is 3 for cubic B-splines
        
    Returns
    -------
    np.ndarray
        Array of length nsp containing integral coefficients
        
    """
    knots = gen_knots(nsp, b_order)
    splines = np.zeros((nsp))
    for i in range(nsp):
        bfunc = BSpline.basis_element(knots[i],extrapolate=False)
        b = np.nan_to_num(bfunc(bvar),copy=False,nan=0.0)
        splines[i] = np.einsum('i,i,i->', b, func, weights)

    return splines
