"""
Code to load pianoroll data (.pickle)
"""
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split

try:
    # Python 3
    import _pickle as cPickle
except:
    # Python 2
    import cPickle

try:
    # Python 2
    range
except:
    # Python 3
    def range(x): return iter(range(x))

rel_keys = {'a': 'C',
            'b-': 'D-',
            'b': 'D',
            'c': 'E-',
            'c#': 'E',
            'd-': 'F-',
            'd': 'F',
            'd#': 'F#',
            'e-': 'G-',
            'e': 'G',
            'f': 'A-',
            'f#': 'A',
            'g': 'B-',
            'g#': 'B',
            'a-': 'C-',
            }


def relative_major(k):
    return k if k.isupper() else rel_keys[k]


def pianoroll_to_song(roll, offset=21):
    f = lambda x: (np.where(x)[0] + offset).tolist()
    return [f(s) for s in roll]


def song_to_pianoroll(song, offset=21):
    """
    song = [(60, 72, 79, 88), (72, 79, 88), (67, 70, 76, 84), ...]
    """
    rolls = []
    all_notes = [y for x in song for y in x]
    if min(all_notes) - offset < 0:
        offset -= 12
        # assert(False)
    if max(all_notes) - offset > 87:
        offset += 12
        # assert(False)
    for notes in song:
        roll = np.zeros(88)
        roll[[n - offset for n in notes]] = 1.
        rolls.append(roll)
    return np.vstack(rolls)


def sliding_inds(n, seq_length, step_length):
    return np.arange(n - seq_length, step=step_length)


def sliding_window(roll, seq_length, step_length=1):
    """
    returns [n x seq_length x 88]
        if step_length == 1, then roll[i,1:] == roll[i+1,:-1]
    """
    rolls = []
    for i in sliding_inds(roll.shape[0], seq_length, step_length):
        rolls.append(roll[i:i + seq_length, :])
    if len(rolls) == 0:
        return np.array([])
    return np.dstack(rolls).swapaxes(0, 2).swapaxes(1, 2)


def songs_to_pianoroll(songs, seq_length, step_length,
                       inner_fcn=song_to_pianoroll):
    """
    songs = [song1, song2, ...]
    """
    rolls = [sliding_window(inner_fcn(s), seq_length, step_length)
             for s in songs]
    rolls = [r for r in rolls if len(r) > 0]
    inds = [i * np.ones((len(r),)) for i, r in enumerate(rolls)]
    return np.vstack(rolls), np.hstack(inds)


class PianoData:

    def __init__(self, train_file, batch_size=None, seq_length=1, step_length=1, return_label_next=True, return_label_hist=False, squeeze_x=True, squeeze_y=True, use_rel_major=True):
        """
        returns [n x seq_length x 88] where rows referring to the same song will overlap an amount determined by step_length

        specifying batch_size will ensure that that mod(n, batch_size) == 0
        """
        try:
            # Python 3
            with open(train_file, 'rb') as pickle_file:
                D = cPickle.load(pickle_file)
        except:
            # Python 2
            with open(train_file) as pickle_file:
                D = cPickle.load(pickle_file)

        self.train_file = train_file  # .pickle source file
        self.batch_size = batch_size  # ensures that nsamples is divisible by this
        self.seq_length = seq_length  # returns [n x seq_length x 88]
        self.step_length = step_length  # controls overlap in rows of X
        # if True, y is next val of X; else y == X
        self.return_label_next = return_label_next
        # if True, y is next val of X for each column of X; else y == [n x 1 x
        # 88]
        self.return_label_hist = return_label_hist
        self.squeeze_x = squeeze_x  # remove singleton dimensions in X?
        self.squeeze_y = squeeze_y  # remove singleton dimensions in y?
        # minor keys get mapped to their relative major, e.g. 'a' -> 'C'
        self.use_rel_major = use_rel_major

        # sequences with song indices
        self.data_train, self.labels_train, self.train_song_inds = self.make_xy(D[
                                                                                'train'])
        self.data_test, self.labels_test, self.test_song_inds = self.make_xy(D[
                                                                             'test'])
        self.data_valid, self.labels_valid, self.valid_song_inds = self.make_xy(D[
                                                                                'valid'])

        # # song index per sequence
        # self.train_song_inds = self.song_inds(D['train'])
        # self.test_song_inds = self.song_inds(D['test'])
        # self.valid_song_inds = self.song_inds(D['valid'])

        # mode per sequence
        if 'train_mode' in D:
            self.train_song_modes = self.song_modes(
                D['train_mode'], self.train_song_inds)
            self.test_song_modes = self.song_modes(
                D['test_mode'], self.test_song_inds)
            self.valid_song_modes = self.song_modes(
                D['valid_mode'], self.valid_song_inds)
        if 'train_key' in D:
            D = self.update_keys(D)
            self.key_map = self.make_keymap(D)
            self.train_labels = self.song_keys(
                D['train_key'], self.train_song_inds)
            self.test_labels = self.song_keys(
                D['test_key'], self.test_song_inds)
            self.valid_labels = self.song_keys(
                D['valid_key'], self.valid_song_inds)

        if seq_length > 1:
            X = vstack([self.data_train,
                        self.data_valid,
                        self.data_test,
                        self.labels_train,
                        self.labels_valid,
                        self.labels_test
                        ])

            idx = X.sum(axis=0).sum(axis=0) > 0

            n_train = self.data_train.shape[0]
            n_valid = self.data_valid.shape[0]
            n_test = self.data_test.shape[0]

            self.data_train = self.data_train[:, :, idx].reshape((n_train, -1))
            self.data_valid = self.data_valid[:, :, idx].reshape((n_valid, -1))
            self.data_test = self.data_test[:, :, idx].reshape((n_test, -1))

            self.labels_train = self.labels_train[:, :, idx]
            self.labels_train = self.labels_train.reshape((n_train, -1))
            self.labels_valid = self.labels_valid[:, :, idx]
            self.labels_valid = self.labels_valid.reshape((n_valid, -1))
            self.labels_test = self.labels_test[
                :, :, idx].reshape((n_test, -1))

            self.original_dim = idx.sum() * seq_length
        else:
            self.original_dim = None

    def make_xy(self, songs):
        inner_fcn = song_to_pianoroll
        data_rolls, song_inds = songs_to_pianoroll(
            songs, self.seq_length + int(self.return_label_next), self.step_length, inner_fcn=inner_fcn)
        data_rolls = self.adjust_for_batch_size(data_rolls)
        song_inds = self.adjust_for_batch_size(song_inds)
        if self.return_label_next:  # make Y the last col of X
            if self.return_label_hist:
                labels_rolls = data_rolls[:, 1:, :]
            else:
                labels_rolls = data_rolls[:, -1, :]
            data_rolls = data_rolls[:, :-1, :]
        else:
            labels_rolls = data_rolls
        if self.squeeze_x:  # e.g., if X is [n x 1 x 88]
            data_rolls = data_rolls.squeeze()
        if self.squeeze_y:
            labels_rolls = labels_rolls.squeeze()
        return data_rolls, labels_rolls, song_inds

    def song_modes(self, modes, song_inds):
        return np.array(modes)[song_inds.astype(int)]

    def update_keys(self, D):
        if not self.use_rel_major:
            return
        D['train_key'] = [relative_major(k) for k in D['train_key']]
        D['test_key'] = [relative_major(k) for k in D['test_key']]
        D['valid_key'] = [relative_major(k) for k in D['valid_key']]
        return D

    def make_keymap(self, D):
        all_keys = np.unique(
            np.hstack([D['train_key'], D['test_key'], D['valid_key']]))
        return dict(zip(all_keys, range(len(all_keys))))

    def song_keys(self, keys, song_inds):
        """
        also converts keys to ints, e.g., ['A', 'B', 'C'] -> [0, 1, 2]
        """
        key_inds = [self.key_map[k] for k in keys]
        return np.array(key_inds)[song_inds.astype(int)]

    def adjust_for_batch_size(self, items):
        if self.batch_size is None:
            return items
        mod = (items.shape[0] % self.batch_size)
        return items[:-mod] if mod > 0 else items


class MNISTData(object):

    def __init__(self, batch_size):
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

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

        self.labels_train = self.data_train
        self.labels_valid = self.data_valid


class ExoplanetData(object):
    exoplanet_filename = 'exoplanet_spectral_database_normalized.joblib.save'
    default_train_file = os.environ['HOME'] + '/.vaelstmpredictor/data/'

    def __init__(self, train_file=None, batch_size=128, test_size=0.30,
                 normalize_spec=False, skip_features=5):
        ''' set skip_features to 0 to use `all` of the data
        '''

        if train_file is None or not os.path.exists(train_file):
            print('[WARNING] `train_file` does not exist. '
                  'Using default location')
            print('[WARNING] `train_file`: {}'.format(train_file))
            print('[WARNING] `default_train_file`: {}'.format(
                self.default_train_file))

            train_file = self.default_train_file

        exoplanet_filename = '{}/{}'.format(train_file,
                                            self.exoplanet_filename)

        spectra, physics = joblib.load(exoplanet_filename)

        if skip_features:
            spectra = spectra[:, ::skip_features]

        idx_train, idx_test = train_test_split(np.arange(len(spectra)),
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

        # This is a 'copy' because the output must equal the input
        self.labels_train = self.data_train
        self.labels_valid = self.data_valid


def info_message(message, end='\n'):
    print('[INFO] {}'.format(message), end=end)


def test_data_shapes(batch_size=128):
    # Compare mnist data shapes to exospec data shapes
    #   to confirm that exospec data is formated 'the same' as mnist data
    #   so far so good!
    from keras.utils import to_categorical
    # Load MNIST Data
    mnistdata = MNISTData(batch_size=batch_size)
    exospec = ExoplanetData(batch_size=batch_size)

    mnistdata.train_labels = to_categorical(mnistdata.train_labels)
    mnistdata.valid_labels = to_categorical(mnistdata.valid_labels)

    print()
    info_message('train_labels')
    print(mnistdata.train_labels.shape)
    print(exospec.train_labels.shape)
    print()

    info_message('valid_labels')
    print(mnistdata.valid_labels.shape)
    print(exospec.valid_labels.shape)
    print()

    info_message('test_labels')
    print(mnistdata.test_labels.shape)
    print(exospec.test_labels.shape)
    print()

    info_message('data_train')
    print(mnistdata.data_train.shape)
    print(exospec.data_train.shape)
    print()

    info_message('data_valid')
    print(mnistdata.data_valid.shape)
    print(exospec.data_valid.shape)
    print()

    info_message('data_test')
    print(mnistdata.data_test.shape)
    print(exospec.data_test.shape)
    print()

    info_message('labels_train')
    print(mnistdata.labels_train.shape)
    print(exospec.labels_train.shape)
    print()

    info_message('labels_valid')
    print(mnistdata.labels_valid.shape)
    print(exospec.labels_valid.shape)


class BlankClass(object):

    def __init__(self):
        pass

if __name__ == '__main__':
    # train_file = '../data/input/Piano-midi_all.pickle'
    train_file = '../data/input/JSB Chorales_all.pickle'
    P = PianoData(train_file, seq_length=1, step_length=1)
