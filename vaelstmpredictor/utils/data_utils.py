"""
Code to load pianoroll data (.pickle)
"""
import numpy as np
import pandas as pd
import os
import joblib

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

try:
    # Python 3
    import _pickle as cPickle
except:
    # Python 2
    import cPickle

def debug_message(message, end='\n'):
    print('[DEBUG] {}'.format(message), end=end)

def info_message(message, end='\n'):
    print('[INFO] {}'.format(message), end=end)

def warning_message(message, end='\n'):
    print('[WARNING] {}'.format(message), end=end)

class MNISTData(object):

    def __init__(self, batch_size):
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        n_samples_test = x_test.shape[0]
        n_samples_test = (n_samples_test // batch_size) * batch_size

        x_test = x_test[:n_samples_test]
        y_test = y_test[:n_samples_test]

        n_samples_train = x_train.shape[0]
        n_samples_train = (n_samples_train // batch_size) * batch_size

        x_train = x_train[:n_samples_train]
        y_train = y_train[:n_samples_train]

        self.image_size = x_train.shape[1]
        self.original_dim = self.image_size * self.image_size

        x_train = np.reshape(x_train, [-1, self.original_dim])
        x_test = np.reshape(x_test, [-1, self.original_dim])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        """These are all of the necessary `data_instance` components"""
        self.train_labels = y_train
        self.valid_labels = y_test
        self.test_labels = np.arange(0)  # irrelevant(?)

        self.data_train = x_train
        self.data_valid = x_test
        self.data_test = np.arange(0)  # irrelevant(?)


class bostonHousingData(object):

    def __init__(self, batch_size):
        from keras.datasets import boston_housing

        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

        # n_samples_test = x_test.shape[0]
        # n_samples_test = (n_samples_test // batch_size) * batch_size

        # x_test = x_test[:n_samples_test]
        # y_test = y_test[:n_samples_test]

        # n_samples_train = x_train.shape[0]
        # n_samples_train = (n_samples_train // batch_size) * batch_size

        # x_train = x_train[:n_samples_train]
        # y_train = y_train[:n_samples_train]

        """These are all of the necessary `data_instance` components"""
        self.train_labels = y_train
        self.valid_labels = y_test
        self.test_labels = np.arange(0)  # irrelevant(?)

        self.data_train = x_train
        self.data_valid = x_test
        self.data_test = np.arange(0)  # irrelevant(?)


class dummyData(object):

    def __init__(self, batch_size):
        n_samples = batch_size*100
        min_amp = 1e-10
        max_amp = 1e3
        amps = np.random.uniform(min_amp, max_amp, n_samples) # wave amplitudes
        phases = np.random.uniform(0, 2*np.pi, n_samples) # wave phase
        periods = np.random.uniform(0.9, 1.1, n_samples) # wave period 

        n_features = 1000
        times = np.linspace(-2*np.pi, 2*np.pi, n_features)
        features = amps[:,None]*np.sin(2*np.pi /periods[:,None] * (times - phases[:,None]))
        print(features.shape)
        labels = np.c_[amps, phases, periods]  # this may need to be `np.r_` instead of `np.c_` (?)

        (x_train, x_test, y_train, y_test) = train_test_split(features, labels, test_size=0.2)

        self.train_labels = y_train
        self.valid_labels = y_test
        self.test_labels = np.arange(0)  # irrelevant(?)

        self.data_train = x_train.astype('float32') / 1000
        self.data_valid = x_test.astype('float32') / 1000
        self.data_test = np.arange(0)  # irrelevant(?)


class dummyData2(object):

    def __init__(self, batch_size, n_features, n_windows, n_steps):
        n_samples = batch_size*100

        features = np.random.normal(0, 1, size=(n_samples, n_features))
        labels = np.random.normal(0, 1, size=(n_samples, 1))

        features_split = np.array([features[0:(n_windows), :]])
        labels_split = np.array([labels[0:(n_windows)]])
        for i in range(1, n_samples, n_steps):
            if(i+n_windows < len(features)):
                fsplit = features[i:(i+n_windows), :]
                features_split = np.r_[features_split,[fsplit]]

                lsplit = labels[i:(i+n_windows)]
                labels_split = np.r_[labels_split,[lsplit]]

        (x_train, x_test, y_train, y_test) = train_test_split(features_split, labels_split, test_size=0.2)

        self.train_labels = np.reshape(y_train, (y_train.shape[0], n_windows))
        self.valid_labels = np.reshape(y_test, (y_test.shape[0], n_windows))

        self.data_train = x_train
        self.data_valid = x_test

class SpitzerCalOrig(object):

    def __init__(self, n_windows, n_steps):
        keep_cols = ['xpos', 'ypos', 'xfwhm', 'yfwhm']
                     #, 'bg_flux', 'pix1', 'pix2', 'pix3', 'pix4', 'pix5', 'pix6',
                     #'pix7', 'pix8', 'pix9', 'fluxerr', 'sigma_bg_flux'] 

        pmap_filename = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv'
        default_train_file = os.environ['HOME'] + '/.vaelstmpredictor/data/'

        df = pd.read_csv(default_train_file+pmap_filename)
        df.dropna(inplace=True)

        # adjust the pixel values to be normalized
        # by the mean of the sum of all pixel values
        df['pixsum'] = df['pix1'] + df['pix2'] + df['pix3'] + df['pix4'] + df['pix5'] + df['pix6'] + df['pix7'] + df['pix8'] + df['pix9']
        medflux = np.median(df['pixsum'])
        meanflux = np.mean(df['pixsum'])
        df['pix1'] /= meanflux
        df['pix2'] /= meanflux
        df['pix3'] /= meanflux
        df['pix4'] /= meanflux
        df['pix5'] /= meanflux
        df['pix6'] /= meanflux
        df['pix7'] /= meanflux
        df['pix8'] /= meanflux
        df['pix9'] /= meanflux

        # normalize bg_flux in some way as well.
        meanbgflux = np.mean(df['bg_flux'])
        df['bg_flux'] /= meanbgflux

        # normalize xerr & yerr in some way as well
        meanxerr = np.mean(df['xerr'])
        meanyerr = np.mean(df['yerr'])
        df['xerr'] /= meanxerr
        df['yerr'] /= meanyerr

        # choose feature set 16 feature feature set
        features = np.array(df[keep_cols])

        # normalize y
        labels = df['flux'] / np.median(df['flux'])
        # labels = np.random.uniform(0.5, 1.5, len(df))
        labels = np.array(labels).reshape((-1, 1)) 

        features_split = features[:(len(features)//n_windows)*n_windows, :].reshape((-1, n_windows, features.shape[1])) 
        labels_split = labels[:(len(labels)//n_windows)*n_windows, :].reshape((-1, n_windows, 1))

        for i in range(1, n_windows, n_steps):
            f_split = features[i:((len(features)-i)//n_windows)*n_windows +i, :].reshape((-1, n_windows, features.shape[1])) 
            l_split = labels[i:((len(labels)-i)//n_windows)*n_windows +i, :].reshape((-1, n_windows, 1))

            features_split = np.r_[features_split, f_split]
            labels_split = np.r_[labels_split, l_split]

        (x_train, x_test, y_train, y_test) = train_test_split(features_split, labels_split, test_size=0.2)

        self.train_labels = np.reshape(y_train, (y_train.shape[0], n_windows))
        self.valid_labels = np.reshape(y_test, (y_test.shape[0], n_windows))

        self.data_train = x_train
        self.data_valid = x_test


def generator(features, labels, lookback, delay, min_index, max_index,
          shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(labels) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           features.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = features.iloc[indices].values
            targets[j] = labels.iloc[rows[j] + delay]
        yield samples, targets

def preprocess_spitzercal(pmap_filename=None, 
                          include_pld=False, 
                          include_bg_flux=False,
                          include_uncs=False):

    default_train_file = os.environ['HOME'] + '/.vaelstmpredictor/data/'

    if pmap_filename is None:
        pmap_filename = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv'

    df = pd.read_csv(os.path.join(default_train_file, pmap_filename))
    df.dropna(inplace=True)

    keep_cols = ['xpos', 'ypos', 'xfwhm', 'yfwhm']

    if include_pld:
        # adjust the pixel values to be normalized
        # by the mean of the sum of all pixel values
        df['pixsum'] = (df['pix1'] + df['pix2'] + df['pix3'] + df['pix4'] + 
                        df['pix5'] + df['pix6'] + df['pix7'] + df['pix8'] + 
                        df['pix9'])

        medflux = np.median(df['pixsum'])
        meanflux = np.mean(df['pixsum'])
        df['pix1'] /= meanflux
        df['pix2'] /= meanflux
        df['pix3'] /= meanflux
        df['pix4'] /= meanflux
        df['pix5'] /= meanflux
        df['pix6'] /= meanflux
        df['pix7'] /= meanflux
        df['pix8'] /= meanflux
        df['pix9'] /= meanflux

        keep_cols.extend(['pix1', 'pix2', 'pix3', 'pix4', 'pix5', 
                          'pix6', 'pix7', 'pix8', 'pix9'])
    
    #, 'bg_flux', , 'fluxerr', 'sigma_bg_flux'] 

    if include_bg_flux:
        # normalize bg_flux in some way as well.
        meanbgflux = np.mean(df['bg_flux'])
        df['bg_flux'] /= meanbgflux

        keep_cols.append('bg_flux')

    if include_uncs:
        # normalize xerr & yerr in some way as well
        meanxerr = np.mean(df['xerr'])
        meanyerr = np.mean(df['yerr'])
        df['xerr'] /= meanxerr
        df['yerr'] /= meanyerr

        keep_cols.extend(['xerr', 'yerr'])

    # choose feature set 16 feature feature set
    features = df[keep_cols]

    # normalize y
    labels = df['flux'] / np.median(df['flux'])
    # labels = np.random.uniform(0.5, 1.5, len(df))
    # labels = np.array(labels).reshape((-1, 1))

    return features, labels

class SpitzerCal(object):

    def __init__(self, lookback=1440, delay=144, step=6, 
                 batch_size=128, test_size=0.2, shuffle=True):
        features, labels = preprocess_spitzercal(pmap_filename=None, 
                                                 include_pld=False, 
                                                 include_bg_flux=False,
                                                 include_uncs=False)

        # (x_train, x_test, y_train, y_test) = train_test_split(
  #           features, labels, test_size=test_size)

        self.train_labels = None
        self.valid_labels = None

        self.data_train = None
        self.data_valid = None

        train_min_idx = 0
        train_max_idx = int((1-test_size) * labels.size)

        test_min_idx = train_max_idx + 1
        test_max_idx = len(labels) - delay - 1
        
        self.train_gen = generator(features, labels,
                              lookback=lookback,
                              delay=delay,
                              min_index=train_min_idx,
                              max_index=train_max_idx,
                              shuffle=shuffle,
                              step=step, 
                              batch_size=batch_size)

        self.val_gen = generator(features, labels,
                            lookback=lookback,
                            delay=delay,
                            min_index=test_min_idx,
                            max_index=test_max_idx,
                            step=step,
                            batch_size=batch_size)

        self.val_steps = (test_max_idx - test_min_idx - lookback) // batch_size

        self.output_shape = (lookback//step)
        self.input_shape = ((lookback//step), features.shape[1])


class ExoplanetData(object):
    exoplanet_filename = 'exoplanet_spectral_folded_database.joblib.save'
    # exoplanet_filename = 'exoplanet_spectral_folded_database_normalized.joblib.save'
    default_train_file = os.environ['HOME'] + '/.vaelstmpredictor/data/'

    exoplanet_data_key = '1KIEDaGkDlcgZmL6t8rDlCp9PGN7glbWq'
    exoplanet_data_online = 'https://drive.google.com/open?id={}'
    exoplanet_data_online = exoplanet_data_online.format(exoplanet_data_key)

    def __init__(self, train_file=None, batch_size=128, test_size=0.20,
                 normalize_spec=False, skip_features=5, use_all_data=True):
        ''' set skip_features to 0 to use `all` of the data
        '''

        if train_file is None:
            info_message('`default_train_file`: {}'.format(
                self.default_train_file))

            train_file = self.default_train_file
        elif not os.path.exists(train_file):
            warning_message('`train_file` does not exist. '
                            'Using default location')

            warning_message('`train_file`: {}'.format(train_file))
            warning_message('`default_train_file`: {}'.format(
                self.default_train_file))

            train_file = self.default_train_file

        exoplanet_filename = '{}/{}'.format(train_file,
                                            self.exoplanet_filename)

        if not os.path.exists(exoplanet_filename):
            info_message('{} does not exist; '
                         'give me data or give me death'.format(train_file))
            info_message('Downloading Exoplanet Spectral Database')
            print('\tThis could several minutes [~15 minutes?]')

            if not os.path.exists(train_file):
                os.mkdir(os.environ['HOME'] + '/.vaelstmpredictor')
                os.mkdir(self.default_train_file)

            self.download_exoplanet_data()

        spectra, physics = joblib.load(exoplanet_filename)

        # if skip_features:
        #     spectra_ = spectra[:, ::skip_features]

        #     if use_all_data:
        #         physics_ = physics.copy()
        #         for k in range(1, skip_features):
        #             debug_message('Physics: {}'.format(physics.shape))
        #             debug_message('Spectra_: {}'.format(spectra_.shape))
        #             debug_message('Spectra: {}'.format(spectra.shape))

        #             physics_ = np.r_[physics_, physics]
        #             spectra_ = np.r_[spectra_, spectra[:, k::skip_features]]

        #     spectra = spectra_

        # idx_shuffle = np.random.shuffle(np.arange(physics.shape[0]))

        # spectra = spectra[idx_shuffle]
        # physics = physics[idx_shuffle]

        idx_train, idx_test = train_test_split(np.arange(physics.shape[0]),
                                               test_size=test_size)

        if normalize_spec:
            for k, spec in enumerate(spectra):
                spectra[k] = spec - np.median(spec)

        x_train = physics[idx_train]  # features for decoder
        y_train = spectra[idx_train]  # labels for decoder

        x_test = physics[idx_test]  # features for decoder
        y_test = spectra[idx_test]  # labels for decoder

        n_samples_test = y_test.shape[0]
        n_samples_test = (n_samples_test // batch_size) * batch_size

        y_test = y_test[:n_samples_test]
        x_test = x_test[:n_samples_test]

        n_samples_train = y_train.shape[0]
        n_samples_train = (n_samples_train // batch_size) * batch_size

        y_train = y_train[:n_samples_train]
        x_train = x_train[:n_samples_train]

        # these are our "labels"; the regresser will be conditioning on these
        
        self.train_labels = x_train
        self.valid_labels = x_test
        self.test_labels = np.array([])  # irrelevant(?)

        # these are our "features"; the VAE will be reproducing these
        self.data_train = y_train
        self.data_valid = y_test
        self.data_test = np.array([])  # irrelevant(?)


    def download_exoplanet_data(self):
        # os.system("git clone https://github.com/jeroenmeulenaar/python3-mega.git")
        # os.system(os.environ['HOME']+"/anaconda3/envs/tf_gpu/bin/pip install -r python3-mega/requirements.txt")
        import subprocess

        pip = os.environ['HOME'] + '/anaconda3/envs/tf_env/bin/pip'
        git = 'git+https://github.com/jeroenmeulenaar/python3-mega.git'
        command = pip + ' install ' + git

        prog = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = prog.communicate()

        from mega import Mega

        mega = Mega()
        m = Mega.from_ephemeral()
        print("Downaling File...")
        # Download normalized data
        m.download_from_url(
            'https://mega.nz/#!O6A3wS4a!vTsMQZxl3VnbPbbksfJ0243oWDxv5z1g1zy4XIow4HQ')
        # os.system('mv exoplanet_spectral_database_normalized.joblib.save '+os.environ['HOME']+'/.vaelstmpredictor/data')

        spec_file_name = 'exoplanet_spectral_folded_database_normalized.joblib.save'
        os.rename(spec_file_name, self.default_train_file + spec_file_name)
        # m.download_from_url('https://mega.nz/#!T7YjkayK!rLqsthYpbbN9dv2yAM6kkjt986soX0KaKsmEqdHeR3U')
        # #Download not normalized data

        # spec_file_name = 'exoplanet_spectral_folded_database.joblib.save'
        # os.rename(spec_file_name, self.default_train_file + spec_file_name)

class BlankClass(object):
    def __init__(self):
        pass
