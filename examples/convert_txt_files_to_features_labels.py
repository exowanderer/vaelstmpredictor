import joblib
import numpy as np

from glob import glob
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
    if wavelengths is None: wavelengths = info_now[:,0]
    if waves_use is None: waves_use = wavelengths < 5.0
    
    goyal_grid_dict[key] = info_now[:,1][waves_use]

''' Organize input data for autoencoder '''
print("[INFO] Assigning input values onto `labels` and `features`")

n_waves = waves_use.sum()
features = np.zeros((len(goyal_filenames), n_waves))
labels = np.zeros((len(goyal_filenames), len(key.split('_'))))

# This config is setup as the predictor archicture
for k, (key,val) in enumerate(goyal_grid_dict.items()): 
    features[k] = val  # Store the spectrum as the feature

    # Store the physical properties as the labels
    labels[k] = np.array(key.split('_')).astype(float)

save_filename = 'exoplanet_spectral_database.joblib.save'

print('[INFO] Saving features and albels to ' + save_filename)
joblib.dump([features, labels], save_filename)
