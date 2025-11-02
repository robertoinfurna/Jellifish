import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.expanduser('~/Desktop/Pangal'))

from pangal.spectrum import Spectrum

from scipy.constants import c, parsec
import astropy.io.fits as fits
import itertools
import argparse
import copy, glob, re, os
import tempfile
from tqdm import tqdm

from pcigale.sed import SED
from pcigale.sed import utils
from pcigale.sed_modules import get_module

def truncated_boissier_sfh(V, R, Q_age, tau_Q, burst_factor, data):

    # --- Select model data ---
    v_array = np.array(list(data.keys()))
    closest_v = v_array[np.argmin(np.abs(v_array - V))]
    #print("Closest velocity:", closest_v)

    r_array = np.array(list(data[closest_v].keys()))
    closest_r = r_array[np.argmin(np.abs(r_array - R))]
    #print("Closest radius:", closest_r)

    selected_data = data[closest_v][closest_r]
    
    AGE = 13500  # Myr    

    # Parameters
    t_trunc = AGE - Q_age    # truncation age
    burst_duration = 10.0    # duration of burst [Myr]
    t_burst_start = t_trunc - burst_duration
    t_burst_end = t_trunc


    # --- CIGALE SFH grid setup ---
    t_myr = selected_data[:, 0] * 1000.0
    sfr_myr = selected_data[:, 2]
    full_time_grid = np.arange(0, AGE + 1)

    interp = np.interp(full_time_grid, t_myr, sfr_myr, left=0.0, right=0.0)

    # --- Apply burst (rectangular increase) ---
    sfr_with_burst = interp.copy()
    burst_mask = (full_time_grid >= t_burst_start) & (full_time_grid < t_burst_end)
    sfr_with_burst[burst_mask] *= (1+burst_factor)

    # --- Apply exponential truncation ---
    sfr_trunc = sfr_with_burst.copy()
    sfr_ttrunc = np.interp(t_trunc, full_time_grid, sfr_with_burst)
    mask_trunc = full_time_grid > t_trunc
    sfr_trunc[mask_trunc] = sfr_ttrunc * np.exp(-(full_time_grid[mask_trunc] - t_trunc) / tau_Q)

    return full_time_grid, sfr_trunc


def truncated_boissier_sed(V, R, Q_age, tau_Q, burst_factor, metallicity, data, AGE=13500):
    
    full_time_grid, sfr_trunc = truncated_boissier_sfh(V, R, Q_age, tau_Q, burst_factor, data=data)

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        for ti, si in zip(full_time_grid, sfr_trunc):
            tmp.write(f"{int(ti):d}   {si:.6e}\n".encode())

    galaxy = SED()
    sfh_module = get_module('sfhfromfile', filename=tmp.name, sfr_column=1, age=AGE, normalise=True)
    ssp_module = get_module('cb19', imf=1, metallicity=metallicity)

    sfh_module.process(galaxy)
    ssp_module.process(galaxy)
    os.remove(tmp.name)

    wl = galaxy.wavelength_grid * 10
    resolution = _compute_resolution(wl, 'cb19')
    stellar_luminosity = galaxy.luminosities['stellar.young'] + galaxy.luminosities['stellar.old']
    stellar_flux_cgs = utils.luminosity_to_flux(stellar_luminosity, 10. * parsec) * 100
    frac_young = galaxy.luminosities['stellar.young'] / stellar_luminosity

    spec = Spectrum(wl=wl, resolution=resolution)
    spec.flux_young = stellar_flux_cgs * frac_young
    spec.flux_old = stellar_flux_cgs * (1 - frac_young)
    spec.flux = stellar_flux_cgs

    spec.header.update({
        'WUNITS': 'A', 'FUNITS': 'erg/s/cm2/A', 'SFH': 'truncated', 'STARLIB': 'cb19',
        'Q_age': Q_age, 'tau_Q': tau_Q, 'burst': burst_factor, 'metal': metallicity
    })

    return spec

# Creates many models
def build_boissier_SEDs(V, R, SFH_QUENCHING_AGE, SFH_QUENCHING_TAU, BURST_FACTORS, METALLICITIES,
                            data, filename=None):

    combos = list(itertools.product(SFH_QUENCHING_AGE, SFH_QUENCHING_TAU, BURST_FACTORS, METALLICITIES))
    print(f"Generating {len(combos)} models")

    models = []
    for Q_age, tau_Q, burst_factor, metallicity in tqdm(combos, desc="Processing combinations"):
        spec = truncated_boissier_sed(V, R, Q_age, tau_Q, burst_factor, metallicity, data=data)
        models.append(spec)

    if filename:
        save_to_fits(filename, models)
    return models

def _compute_resolution(wl, stellar_library):
    if stellar_library == 'bc03':
        delta_lambda = wl / 300
        delta_lambda[(wl >= 3200) & (wl <= 9500)] = 3.0
    elif stellar_library == 'cb19':
        delta_lambda = np.full_like(wl, 2.0)
        delta_lambda[(wl >= 912) & (wl <= 3540)] = 1.0
        delta_lambda[(wl >= 3540) & (wl <= 7350)] = 2.5
        delta_lambda[(wl >= 7350) & (wl <= 9400)] = 1.0
    else:
        print(f"WARNING: Unknown library {stellar_library}, using bc03 resolution.")
        delta_lambda = wl / 300
        delta_lambda[(wl >= 3200) & (wl <= 9500)] = 3.0
    return wl / delta_lambda

def save_to_fits(filename,models):
    
    # Primary HDU: wavelength array as data
    wl_data = models[0].wl.astype(np.float32)
    res_data = models[0].resolution.astype(np.float32)
    primary_hdu = fits.PrimaryHDU(data=[wl_data,res_data])
    primary_hdu.header["NMODEL"] = len(models)  # global metadata
    
    # Each model in its own extension
    model_hdus = []
    for i, spec in enumerate(models):
        cols = [
            fits.Column(name="FLUX",       array=spec.flux,       format="E"),
            fits.Column(name="FLUX_YOUNG", array=spec.flux_young, format="E"),
            fits.Column(name="FLUX_OLD",   array=spec.flux_old,   format="E"),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, name=f"MODEL_{i}")

        # Save model-specific metadata
        for k, v in spec.header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass  # ignore keywords that FITS can't store

        model_hdus.append(hdu)

    hdul = fits.HDUList([primary_hdu] + model_hdus)
    hdul.writeto(filename, overwrite=True)


def load_spectrum_models_from_fits(filename):
    with fits.open(filename) as hdul:
        # Wavelength array from primary HDU
        wl,resolution = hdul[0].data
        nmodels = len(hdul) - 1

        models = []
        for i in range(1, len(hdul)):
            data = hdul[i].data
            header = hdul[i].header


            spec = Spectrum(wl=wl,resolution=resolution,flux=data["FLUX"])

            spec.flux_young = data["FLUX_YOUNG"]
            spec.flux_old   = data["FLUX_OLD"]
            
            # Keep the model's metadata
            spec.header.update(header)

            models.append(spec)

    return models
