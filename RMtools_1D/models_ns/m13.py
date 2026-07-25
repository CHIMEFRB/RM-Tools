# =============================================================================#
#                          MODEL DEFINITION FILE (m13.py)                      #
#        full_fit with systematics (differential phase & response between X,Y) #
#        and non-zero/frequency-dependent linear and circular polarization     # 
# =============================================================================#
import bilby
import numpy as np
from astropy.constants import c

C = c.value


# -----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms         #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
# -----------------------------------------------------------------------------#
def model(pDict, lamSqArr_m2):
    """
    Full model: Faraday thin source + systematics (cable delay + differential response 
    between X,Y polarizations) + frequency-dependent L/I, V/I

    Was named "cable_delay+response" in the previous pipeline 

    Linear polarization:
        p * exp[ 2i (psi0 + RM lambda^2) ]

    I -> Q leakage here. The pDict has 'gain_diff'

    """

    # Frequency array
    freqArr = C / np.sqrt(lamSqArr_m2)

    # Fractional linear polarization
    pfracArr = (pDict["fracPol"] * np.ones_like(freqArr)) * (freqArr / 400e6)**pDict['gamma']
    pArr = pfracArr * np.ones_like(lamSqArr_m2)

    vfracArr = (pDict['fracPol_V'] * np.ones_like(lamSqArr_m2)) * (freqArr / 400e6)**pDict['gamma_V']    

    gain_X = 1.0 
    gain_Y = gain_X * pDict['gain_diff']

    # Intrinsic Faraday rotation
    quArr = pArr * np.exp( 2j * (np.radians(pDict["psi0_deg"]) +
			         pDict["RM_radm2"] * lamSqArr_m2)  )

    qArr = quArr.real
    uArr = quArr.imag
    vArr = vfracArr * np.ones_like(lamSqArr_m2)

    # Differential phase between X/Y feeds
    phase = 2 * np.pi * freqArr * pDict["lag_s"] + np.radians(pDict["lag_phi"])

    u_leak = np.cos(phase) * uArr - np.sin(phase) * vArr
    v_leak = np.cos(phase) * vArr + np.sin(phase) * uArr

    uArr = u_leak 
    vArr = -v_leak

    # model the differential X,Y response (see Johnston 2006 for details)
    qArr_leak = 0.5*np.ones_like(lamSqArr_m2)*(gain_X**2-gain_Y**2) + 0.5*qArr*(gain_X**2+gain_Y**2)
    qArr = qArr_leak 
    uArr = uArr*gain_X*gain_Y
    vArr = vArr*gain_X*gain_Y 

    quArr = qArr + 1j*uArr

    return quArr, vArr


# -----------------------------------------------------------------------------#
# Priors for the above model.                                                 #
# See https://lscsoft.docs.ligo.org/bilby/prior.html for details              #
# -----------------------------------------------------------------------------#

prior_config = {
    "fracPol": dict(
        minimum=0.001, maximum=1.1,
        name="fracPol", latex_label=r"$p$",
    ),
    "psi0_deg": dict(
        minimum=0.0, maximum=180.0,
        name="psi0_deg", latex_label=r"$\psi_0$ (deg)",
        boundary="periodic",
    ),
    "RM_radm2": dict(
        minimum=-4000.0, maximum=4000.0,
        name="RM_radm2", latex_label=r"RM (rad m$^{-2}$)",
    ),
    "lag_s": dict(
        minimum=-2e-9, maximum=0,
        name="lag_s", latex_label=r"lag (sec)",
    ),
    "lag_phi": dict(
        minimum=0.0, maximum=360.0,
        name="lag_phi", latex_label=r"lag$_\phi$ (deg)",
        boundary="periodic",
    ),
    "gamma": dict(
        minimum=-10.0, maximum=10.0,
        name='gamma', latex_label=r"$\gamma_L$",
    ),
    "fracPol_V": dict(
        minimum=-1.0, maximum=1.0,
        name='fracPol_V', latex_label=r"$p_V$",
    ),
    "gamma_V": dict(
        minimum=-10.0, maximum=10.0,
        name='gamma_V', latex_label=r"$\gamma_V$",
    ),
    "gain_diff": dict(
        minimum=0.1, maximum=10.0,
        name='gain_diff', latex_label=r"gain diff",
    ),
}

def get_priors(bounds=None):
    """
    Building the bilby prior dictionary for the model, with the flexibility of specifying
    different user-input bounds where desired. Can just specify the bounds for the parameter(s) you want;
    e.g., if you have 5 parameters for the simple cable delay model and you just want to modify the RM bounds

    The bounds would be something like {'fracPol': [0.0, 1.0], 'RM_radm2': [-200, 200]}
    """
    bounds = bounds or {}
    priors = {}
    for par, config in prior_config.items():
        config = config.copy()
        if par in bounds: #so if a parameter's bounds is manually specified 
            config['minimum'], config['maximum'] = bounds[par]
        priors[par] = bilby.prior.Uniform(**config)
    return priors 

priors = get_priors()
