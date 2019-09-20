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

def write(string):
    f = open("output.txt", "a")
    f.write(str(string)+"\n")
    f.close()

class Chromosome(object):

    def __init__(self, size_filter, size_kernel, size_pool,
                vae_latent_dim, dnn_hidden_dims, vae_hidden_dims, num_conv_layers,
                batch_size=128, num_epochs=50, dropout_rate=0.7,
                dnn_kl_weight=1, dnn_weight=1, vae_kl_weight=1, vae_weight=1,
                optimizer="adam", predictor_type="prediction", train_file="exoplanet",
                log_dir="./logs", model_dir="../data/models", table_dir="../data/tables",
                save_model=False, verbose=True, rainout=False):
        ''' Configure dnn '''
        #network parameters
        self.size_filter = size_filter
        self.size_kernel = size_kernel
        self.size_pool = size_pool
        self.vae_latent_dim = vae_latent_dim
        self.dnn_hidden_dims = dnn_hidden_dims
        self.vae_hidden_dims = vae_hidden_dims
        self.num_conv_layers = num_conv_layers

        self.rainout = rainout
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate

        self.dnn_kl_weight = dnn_kl_weight
        self.dnn_weight = dnn_weight
        self.vae_kl_weight = vae_kl_weight
        self.vae_weight = vae_weight

        self.optimizer = optimizer
        self.predictor_type = predictor_type
        self.train_file = train_file
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.table_dir = table_dir
        self.save_model = save_model

        np.random.seed(42)
        if(train_file == "exoplanet"):
            from vaelstmpredictor.utils.data_utils import ExoplanetData
            data = ExoplanetData(self.batch_size, normalize_spec = True)

            condensation_category_train = data.train_labels[:,0] == int(rainout)
            condensation_category_validation = data.valid_labels[:,0] == int(rainout)

            scaler = MinMaxScaler()
            self.y_train = scaler.fit_transform(data.train_labels[condensation_category_train, 1:])
            self.y_test = scaler.transform(data.valid_labels[condensation_category_validation, 1:])

            self.x_train = data.data_train[condensation_category_train]
            self.x_test = data.data_valid[condensation_category_validation]

        elif(train_file == "dummyData"):
            from vaelstmpredictor.utils.data_utils import dummyData
            data = dummyData(self.batch_size)
            min_val = -1000
            max_val = 1000
            self.x_train = (data.data_train - min_val)/(max_val - min_val)
            self.y_train = data.train_labels
            self.x_test = (data.data_valid - min_val)/(max_val - min_val)
            self.y_test = data.valid_labels

        elif(train_file == "mnist"):
            from keras.datasets import mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

            self.image_size = self.x_train.shape[1]
            self.original_dim = self.image_size * self.image_size
            self.x_train = np.reshape(self.x_train, [-1, self.original_dim])
            self.x_test = np.reshape(self.x_test, [-1, self.original_dim])
            self.x_train = self.x_train.astype('float32') / 255
            self.x_test = self.x_test.astype('float32') / 255

        elif(train_file == "bostonHousing"):
            from vaelstmpredictor.utils.data_utils import bostonHousingData
            data = bostonHousingData(self.batch_size)
            self.x_train = data.data_train
            self.y_train = data.train_labels
            self.x_test = data.data_valid
            self.y_test = data.valid_labels

        else:
            if(self.verbose):
                print("Train File \"{}\" not recognized".format(train_file))
            return
        
        self.original_dim = data.data_train.shape[1]
        self.input_shape = (self.original_dim, )

        if(len(self.y_train.shape) == 1):
            self.dnn_latent_dim = 1
        else:
            self.dnn_latent_dim = self.y_train.shape[1]
        
        self.VAE()

    # WORKS ONLY IF X DATA IS BETWEEN 0 and 1. 
    def VAE(self):

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        kernel_regularizer = l1_l2(l1 = 0, l2 = 0)

        x = Reshape(self.input_shape+(1,))(inputs)
        #--------- Encoder CNN Layers ------------
        zipper = zip(self.size_filter,
                     self.size_kernel,
                     self.size_pool)

        for i, (fsize, ksize, psize) in enumerate(zipper):
            name = "Encoder_CNN_{}".format(i)
            x = Conv1D(fsize, (ksize,),
                       padding='same',
                       kernel_regularizer=kernel_regularizer,
                       name=name)(x)

            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if(psize != 0):
            	x = MaxPooling1D((psize, ), padding='same')(x)
        #-------------------------------------------
        last_dense_dim = K.int_shape(x)[1]
        x = Flatten()(x)
        #--------- Encoder Dense Layers ------------
        for i, dense_size in enumerate(self.vae_hidden_dims):
            if(i < len(self.vae_hidden_dims) -1):
                x = Dense(dense_size, activation='relu')(x)
            else:
                x = Dense(dense_size, activation='sigmoid')(x)
            # x = Dropout(self.dropout_rate)(x)
        #-------------------------------------------

        self.z_mean_vae = Dense(self.vae_latent_dim, name='z_mean')(x)
        self.z_log_var_vae = Dense(self.vae_latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z_vae = Lambda(sampling, output_shape=(self.vae_latent_dim, ), name='vae_latent_layer')([self.z_mean_vae, self.z_log_var_vae])
        (z_dnn, outputs_dnn) = self.DNN(inputs)

        latent = concatenate([z_vae, z_dnn])

        # instantiate encoder model
        encoder = Model(inputs, latent, name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(K.int_shape(latent)[1], ), name='z_sampling')

        x = latent_inputs
        #--------- Decoder Dense Layers ------------
        for dense_size in self.vae_hidden_dims[::-1]:
            x = Dense(dense_size, activation='relu')(x)
            # x = Dropout(self.dropout_rate)(x)
        #-------------------------------------------

        x = Dense(last_dense_dim, activation='relu')(x)

        x = Reshape((last_dense_dim, 1))(x)
        #--------- Decoder CNN Layers ------------
        zipper = zip(self.size_filter[::-1],
                     self.size_kernel[::-1],
                     self.size_pool[::-1])

        for i, (fsize, ksize, psize) in enumerate(zipper):
            name = "Decoder_CNN_{}".format(i)
            x = Conv1D(fsize, (ksize,),
                       padding='same',
                       kernel_regularizer=kernel_regularizer,
                       name=name)(x)

            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if(psize != 0):
            	x = UpSampling1D(psize)(x)
        #------------------------------------------
        x = Conv1D(1, (3, ), padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = Flatten()(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, x, name='vae_reconstruction')

        # instantiate VAE model
        outputs_vae = decoder(encoder(inputs))

        outputs_total = [outputs_vae, z_dnn, outputs_dnn, z_vae]
        self.model = Model([inputs], outputs_total, name='vae')

    def DNN(self, inputs_dnn):

        kernel_regularizer = l1_l2(l1 = 0, l2 = 0)
        x = Reshape(self.input_shape+(1,))(inputs_dnn)
        #--------- Predictor CNN Layers ------------
        zipper = zip(self.size_filter,
                     self.size_kernel,
                     self.size_pool)

        for i, (fsize, ksize, psize) in enumerate(zipper):
            name = "Predictor_CNN_{}".format(i)
            x = Conv1D(fsize, (ksize,),
                       padding='same',
                       kernel_regularizer=kernel_regularizer,
                       name=name)(x)

            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if(psize != 0):
            	x = MaxPooling1D((psize, ), padding='same')(x)
        #---------------------------------
        x = Flatten()(x)

        #--------- Predictor Dense Layers ------------
        for i, dense_size in enumerate(self.dnn_hidden_dims):
            if(i < len(self.dnn_hidden_dims) -1):
                x = Dense(dense_size, activation='relu')(x)
            else:
                x = Dense(dense_size, activation='sigmoid')(x)
            # x = Dropout(self.dropout_rate)(x)
        #-------------------------------------------

        self.z_mean_dnn = Dense(self.dnn_latent_dim, name='z_mean_dnn')(x)
        self.z_log_var_dnn = Dense(self.dnn_latent_dim, name='z_log_var_dnn')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z_dnn = Lambda(sampling, output_shape=(self.dnn_latent_dim,), name='dnn_latent_layer')([self.z_mean_dnn, self.z_log_var_dnn])
        outputs_dnn = Lambda(lambda x: x + 0, name='dnn_predictor_layer')(z_dnn)
        # instantiate encoder model
        return (z_dnn, outputs_dnn)

    def train(self, verbose=False):
        callbacks = [TerminateOnNaN()]
        callbacks.append(ModelCheckpoint(filepath='conv_dnn_only_weights.hdf5', verbose=self.verbose, save_best_only=True))
        callbacks.append(TensorBoard(log_dir=self.log_dir, histogram_freq=0, batch_size=32, write_graph=True, 
            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
            embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
        callbacks.append(EarlyStopping(patience=20))
        callbacks.append(History())

        vae_weight = 1
        dnn_kl_weight = 1
        dnn_weight = 1
        vae_kl_weight = 1
        self.model.compile(
            optimizer=self.optimizer,

            loss={'vae_reconstruction': self.vae_reconstruction_loss,
                  'dnn_latent_layer': self.dnn_kl_loss,
                  'dnn_predictor_layer': self.dnn_predictor_loss,
                  'vae_latent_layer': self.vae_kl_loss},

            loss_weights={'vae_reconstruction': vae_weight,
                          'dnn_latent_layer': dnn_kl_weight,
                          'dnn_predictor_layer': dnn_weight,
                          'vae_latent_layer': vae_kl_weight},
        )
        if(self.verbose):
            self.model.summary()
        
        self.history = self.model.fit(self.x_train,
                [self.x_train, self.x_train, self.y_train, self.y_train],
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                validation_data=(self.x_test, [self.x_test, self.x_test, self.y_test, self.y_test]),
                callbacks=callbacks)

        best_loss = None
        best_loss_index = None
        for (i,x) in enumerate(self.history.history['val_loss']):
            if(x>0 and (best_loss == None or x<best_loss)):
                best_loss = x
                best_loss_index = i

        self.best_losses = {"val_vae_reconstruction_loss": self.history.history['val_vae_reconstruction_loss'][best_loss_index],
                            "val_vae_latent_layer_loss": self.history.history['val_vae_latent_layer_loss'][best_loss_index],
                            "val_dnn_latent_layer_loss": self.history.history['val_dnn_latent_layer_loss'][best_loss_index],
                            "val_dnn_predictor_layer_loss": self.history.history['val_dnn_predictor_layer_loss'][best_loss_index]}
        self.fitness = 1/best_loss

        if(self.save_model):
            self.model.save_weights('vae_mlp_mnist.h5')

    def vae_kl_loss(self, y_true, y_pred):
        kl_loss = 1 + self.z_log_var_vae - K.square(self.z_mean_vae) - K.exp(self.z_log_var_vae)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return kl_loss

    def dnn_kl_loss(self, y_true, y_pred):
        kl_loss = 1 + self.z_log_var_dnn - K.square(self.z_mean_dnn) - K.exp(self.z_log_var_dnn)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return kl_loss

    def vae_reconstruction_loss(self, y_true, y_pred):
        reconstruction_loss = mean_squared_error(y_true, y_pred)
        # reconstruction_loss *= self.original_dim

        return reconstruction_loss

    def dnn_predictor_loss(self, y_true, y_pred):
        prediction_loss = mean_squared_error(y_true, y_pred)
        return prediction_loss

    def classification_loss(self, y_true, y_pred):
        classification_loss = categorical_crossentropy(y_true, y_pred)
        return classification_loss

if __name__ == '__main__':
    chrom = Chromosome()
    chrom.train()
