import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.expanduser('~/Desktop/Pangal'))
from scipy.interpolate import interp1d
from pangal.spectrum import Spectrum

from scipy.constants import c, parsec
import astropy.io.fits as fits
import itertools
import argparse
import copy, glob, re, os
import tempfile
from tqdm import tqdm

from astropy.io import fits
from pcigale.sed import SED
from pcigale.sed import utils
from pcigale.sed_modules import get_module





### --- LOADS BOISSIER UNPERTURBED MODELS --- ###
# Base directory containing the “DISKEVOL.RES_L0.05_V*” files
data_dir = "SFHs_Boissier/boissier_models_big_grid"

# Pattern to match files without “lignes” in their name
file_pattern = os.path.join(data_dir, "DISKEVOL.RES_L0.05_V*")

# Regex to extract the integer velocity after “_V”
vel_pattern = re.compile(r"_V(\d+)")
# First, collect (velocity, full_path) pairs
velocity_files = []
for full_path in glob.glob(file_pattern):
    fname = os.path.basename(full_path)
    if "lignes" in fname:
        continue
    m = vel_pattern.search(fname)
    if not m:
        continue
    velocity = int(m.group(1))
    velocity_files.append((velocity, full_path))

# Sort by velocity (ascending)
velocity_files.sort(key=lambda vf: vf[0])

# Now build the dictionary in sorted order
boissier_models = {}
for velocity, full_path in velocity_files:
    boissier_models[velocity] = {}   # Initialize nested dict for this velocity

    current_radius = None
    boissier_models_by_radius = {}   # Temporary storage: radius → list of rows

    with open(full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("99999"):
                # Skip empty lines or separator lines
                continue

            if line.startswith("R ="):
                parts = line.split()
                try:
                    # Parse “R = 1.24 kpc” → radius = 1.24
                    current_radius = float(parts[2])
                    boissier_models_by_radius[current_radius] = []
                except (IndexError, ValueError):
                    current_radius = None
                continue

            if current_radius is None:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                # Convert columns 1–5 to float
                row = [float(x) for x in parts[1:6]]
                boissier_models_by_radius[current_radius].append(row)
            except ValueError:
                continue

    # Convert each list of rows into a NumPy array, then store under data[velocity]
    for radius, rows in boissier_models_by_radius.items():
        boissier_models[velocity][radius] = np.array(rows)
        
        
        
        
def unperturbed_boissier_sfh(V, R, t_common=None):

    if t_common is None:
        t_common = np.linspace(0, 13.5, 300)

    V_grid = np.array(sorted(boissier_models.keys()))

    if not (V_grid.min() <= V <= V_grid.max()):
        raise ValueError(f"V={V} outside range")

    sfh_at_V = []

    for Vg in V_grid:

        radius_dict = boissier_models[Vg]
        R_grid = np.array(sorted(radius_dict.keys()))

        """
        if not (R_grid.min() <= R <= R_grid.max()):
            raise ValueError(
                f"R={R} outside range {R_grid.min()}–{R_grid.max()}"
            )
        """

        # --- resample all radii at this V ---
        sfh_resampled = []
        for Rg in R_grid:
            sfh = radius_dict[Rg]
            sfh_resampled.append(resample_sfh(sfh, t_common))

        sfh_resampled = np.stack(sfh_resampled, axis=0)
        # shape: (N_R, N_time, N_cols)

        # --- interpolate in radius ---
        sfh_R = np.array([
            np.interp(R, R_grid, sfh_resampled[:, i, j])
            for i in range(len(t_common))
            for j in range(sfh_resampled.shape[2])
        ]).reshape(len(t_common), -1)

        sfh_at_V.append(sfh_R)

    sfh_at_V = np.stack(sfh_at_V, axis=0)
    # shape: (N_V, N_time, N_cols)

    # --- interpolate in V ---
    sfh_final = np.array([
        np.interp(V, V_grid, sfh_at_V[:, i, j])
        for i in range(len(t_common))
        for j in range(sfh_at_V.shape[2])
    ]).reshape(len(t_common), -1)

    return sfh_final

def resample_sfh(sfh, t_common):
    t = sfh[:, 0]
    out = np.zeros((len(t_common), sfh.shape[1]))
    out[:, 0] = t_common

    for col in range(1, sfh.shape[1]):
        out[:, col] = np.interp(
            t_common, t, sfh[:, col],
            left=sfh[0, col],
            right=sfh[-1, col]
        )
    return out




def truncated_boissier_sfh(V, R, Q_age, tau_Q, burst_factor,):

    
    unperturbed_model = unperturbed_boissier_sfh(V,R)
    
    AGE = 13500  # Myr    

    # Parameters
    t_trunc = AGE - Q_age    # truncation age
    burst_duration = 10.0    # duration of burst [Myr]
    t_burst_start = t_trunc - burst_duration
    t_burst_end = t_trunc


    # --- CIGALE SFH grid setup ---
    t_myr = unperturbed_model[:, 0] * 1000.0
    sfr_myr = unperturbed_model[:, 2]
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


def truncated_boissier_sed(V, R, Q_age, tau_Q, burst_factor, metallicity, AGE=13500):
    
    full_time_grid, sfr_trunc = truncated_boissier_sfh(V, R, Q_age, tau_Q, burst_factor,)

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
def build_boissier_SEDs(V, R, SFH_QUENCHING_AGE, SFH_QUENCHING_TAU, BURST_FACTORS, METALLICITIES, filename=None):

    combos = list(itertools.product(SFH_QUENCHING_AGE, SFH_QUENCHING_TAU, BURST_FACTORS, METALLICITIES))
    print(f"Generating {len(combos)} models")

    models = []
    for Q_age, tau_Q, burst_factor, metallicity in tqdm(combos, desc="Processing combinations"):
        spec = truncated_boissier_sed(V, R, Q_age, tau_Q, burst_factor, metallicity,)
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
