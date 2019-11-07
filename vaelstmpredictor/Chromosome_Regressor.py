import numpy as np
from keras import backend as K
# from keras.callbacks import ProgbarLogger, History, RemoteMonitor, LearningRateScheduler
# from keras.callbacks import CSVLogger, ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard, History
from keras.layers import Input, Lambda, concatenate, Dense
from keras.layers import Conv1D, MaxPooling1D, Layer, Add, Multiply
from keras.layers import UpSampling1D, Flatten, Reshape
from keras.layers import BatchNormalization, Dropout, Activation
from keras.losses import mse, mean_squared_error, binary_crossentropy, categorical_crossentropy
from keras.models import Model, Sequential
from keras.regularizers import l1_l2

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

def debug_message(message, end='\n'):
    print('[DEBUG] {}'.format(message), end=end)


def info_message(message, end='\n'):
    print('[INFO] {}'.format(message), end=end)

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class Chromosome(object):

    def __init__(self, data, size_filter, size_kernel, size_pool,
                dnn_hidden_dims, num_conv_layers,
                batch_size=128, num_epochs=50,
                dropout_rate=0.7, l1_coef = 0.01, l2_coef = 0.01,
                dnn_kl_weight=1, dnn_weight=1,
                optimizer="adam", predictor_type="prediction",
                log_dir="./logs", model_dir="../data/models", table_dir="../data/tables",
                save_model=False, verbose=True, rainout=False):
        ''' Configure dnn '''
        #network parameters
        self.size_filter = size_filter
        self.size_kernel = size_kernel
        self.size_pool = size_pool
        self.dnn_hidden_dims = dnn_hidden_dims
        self.num_conv_layers = num_conv_layers

        self.rainout = rainout
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.l1 = l1_coef
        self.l2 = l2_coef
        self.steps_per_epoch = 500

        self.dnn_kl_weight = dnn_kl_weight
        self.dnn_weight = dnn_weight

        self.optimizer = optimizer
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.table_dir = table_dir
        self.save_model = save_model

        np.random.seed(42)
        self.x_train = data.data_train
        self.y_train = data.train_labels
        self.x_test = data.data_valid
        self.y_test = data.valid_labels
        self.data = data

        '''
        self.original_dim = data.data_train.shape
        self.input_shape = (self.original_dim[1], self.original_dim[2])
        
        if(len(self.y_train.shape) == 1):
            self.dnn_latent_dim = 1
        else:
            self.dnn_latent_dim = self.y_train.shape[1]
        '''

        self.dnn_latent_dim = self.data.output_shape
        self.input_shape = self.data.input_shape
        self.DNN()

    def DNN(self):

        inputs_dnn = Input(shape=self.input_shape, name='inputs_dnn')
        kernel_regularizer = l1_l2(l1 = self.l1, l2 = self.l2)

        x = Reshape(self.input_shape)(inputs_dnn)
        #--------- Predictor CNN Layers ------------
        zipper = zip(self.size_filter,
                     self.size_kernel,
                     self.size_pool)

        for i, (fsize, ksize, psize) in enumerate(zipper):
            cnn_name = "Predictor_{}_{}".format("CNN", i)
            pool_name = "Predictor_{}_{}".format("MaxPool", i)
            x = Conv1D(fsize, (ksize, ),
                       padding='same',
                       kernel_regularizer=kernel_regularizer,
                       name=cnn_name)(x)

            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if(psize != 0):
            	x = MaxPooling1D((psize, ), padding='same', name=pool_name)(x)
        #---------------------------------
        x = Flatten(name="Predictor_Flatten")(x)

        #--------- Predictor Dense Layers ------------
        for i, dense_size in enumerate(self.dnn_hidden_dims):
            dense_name = "Predictor_{}_{}".format("Dense", i)
            if(i < len(self.dnn_hidden_dims) -1):
                x = Dense(dense_size, activation='relu', name=dense_name)(x)
            else:
                x = Dense(dense_size, activation='sigmoid', name=dense_name)(x)
            x = Dropout(self.dropout_rate)(x)
        #-------------------------------------------

        outputs_dnn = Dense(self.dnn_latent_dim, name='outputs_dnn')(x)
        # instantiate encoder model
        self.model = Model(inputs_dnn, outputs_dnn, name='dnn')
        self.model.summary()

    def train(self, verbose=False):
        callbacks = [TerminateOnNaN()]
        if(self.save_model):
        	callbacks.append(ModelCheckpoint(filepath='conv_dnn_only_weights.hdf5', verbose=self.verbose, save_best_only=True))
        callbacks.append(TensorBoard(log_dir=self.log_dir, histogram_freq=0, batch_size=32, write_graph=True, 
            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
            embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
        callbacks.append(EarlyStopping(patience=10))
        callbacks.append(History())

        self.model.compile(
            optimizer=self.optimizer,

            loss={'outputs_dnn': self.dnn_predictor_loss}
        )
        
        if hasattr(self.data, "train_gen"):
	        self.history = self.model.fit_generator(self.data.train_gen,
	                steps_per_epoch=self.steps_per_epoch,
	                epochs=self.num_epochs,
	                validation_data=self.data.val_gen,
	                validation_steps=self.data.val_steps,
	                callbacks=callbacks)
        else:
        	self.history = self.model.fit(self.x_train, self.y_train,
	                epochs=self.num_epochs,
	                batch_size=self.batch_size,
	                validation_data=(self.x_test, self.y_test),
	                callbacks=callbacks)

        best_loss = None
        best_loss_index = None
        for (i,x) in enumerate(self.history.history['val_loss']):
            if(x>0 and (best_loss == None or x<best_loss)):
                best_loss = x
                best_loss_index = i

        self.fitness = 1/best_loss

        if(self.save_model):
            self.model.save_weights('dnn_mlp_mnist.h5')

    def dnn_kl_loss(self, y_true, y_pred):
        kl_loss = 1 + self.z_log_var_dnn - K.square(self.z_mean_dnn) - K.exp(self.z_log_var_dnn)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return kl_loss

    def dnn_predictor_loss(self, y_true, y_pred):
        prediction_loss = mean_squared_error(y_true, y_pred)
        return prediction_loss

    def classification_loss(self, y_true, y_pred):
        classification_loss = categorical_crossentropy(y_true, y_pred)
        return classification_loss

if __name__ == '__main__':
    chrom = Chromosome()
    chrom.train()
