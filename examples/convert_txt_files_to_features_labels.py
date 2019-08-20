import joblib
import numpy as np

from glob import glob
from statsmodels.robust import scale
from tqdm import tqdm
wavelengths = None
waves_use = None

print("[INFO] Load data from harddrive.")

goyal_filenames = glob('localcond_transfiles_txt/trans*.txt.gz')
goyal_filenames.extend(glob('rainout_transfiles_txt/trans*.txt.gz'))

goyal_grid_dict = {}

for fname in tqdm(goyal_filenames):
    key = '_'.join(fname.split('/')[-1].split('_')[1:7])

    # Set the rainout vs local_cond as a boolan categorical variable
    #	i.e the first element of `key` is "is this rainout"
    key = '1_' + key if 'rainout' in fname else '0_' + key

    info_now = np.loadtxt(fname)
    if wavelengths is None:
        wavelengths = info_now[:, 0]
    if waves_use is None:
        waves_use = wavelengths < 5.0

    goyal_grid_dict[key] = info_now[:, 1][waves_use]

''' Organize input data for autoencoder '''
print("[INFO] Assigning input values onto `physics` and `spectra`")

n_waves = waves_use.sum()
spectra = np.zeros((len(goyal_filenames), n_waves))
physics = np.zeros((len(goyal_filenames), len(key.split('_'))))

# This config is setup as the predictor archicture
for k, (key, val) in enumerate(goyal_grid_dict.items()):
    spectra[k] = val  # Store the spectrum as the feature

    # Store the physical properties as the physics
    physics[k] = np.array(key.split('_')).astype(float)

# from Wikipedia: https://en.wikipedia.org/wiki/
#   Median_absolute_deviation#Relation_to_standard_deviation
mad2std = 1.4826  # Create a more stable Std-Dev
spectra_normed = np.zeros(spectra.shape)
for k, spec in enumerate(spectra):
    med_spec = np.median(spec)
    mad_spec = scale.mad(spec) * mad2std
    spectra_normed[k] = (spec - med_spec) / mad_spec

base_save_filename = 'exoplanet_spectral_database.joblib.save'

print('[INFO] Saving spectra and physics to ' + base_save_filename)
joblib.dump([spectra, physics], base_save_filename)


norm_save_filename = 'exoplanet_spectral_database_normalized.joblib.save'

print('[INFO] Saving normalized spectra and physics to ' + norm_save_filename)
joblib.dump([spectra_normed, physics], norm_save_filename)
