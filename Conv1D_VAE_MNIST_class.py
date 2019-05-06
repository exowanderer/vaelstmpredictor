from keras import layers, metrics
from keras import backend as K

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, UpSampling1D
from keras.utils import plot_model

import numpy as np

import keras

def build_hidden_layers(hidden_dims, input_layer, kernel_sizes = None, 
                base_layer_name = '', activation = 'relu', 
                Layer = None, strides=None):
    '''Need to remove all leading zeros for the Decoder 
    to be properly established'''
    if Layer is Dense or kernel_sizes is None:
        return build_hidden_dense_layers(hidden_dims, input_layer, 
                            base_layer_name, activation, Layer = Dense)
    if (Layer is Conv1D or Layer is Conv1D) and kernel_sizes is not None:
        assert(len(kernel_sizes) == len(hidden_dims)), \
            "each Conv layer requires a given kernel_size"
        return build_hidden_conv_layers(filter_sizes, kernel_sizes, strides, 
                input_layer, base_layer_name, activation, Layer = Layer)

def build_hidden_dense_layers(hidden_dims, input_layer, 
                        base_layer_name, activation, Layer = Dense):
    '''Need to remove all leading zeros for the Decoder 
    to be properly established'''

    hidden_dims = list(hidden_dims)
    while 0 in hidden_dims: hidden_dims.remove(0)
    
    # Establish hidden layer structure
    for k, layer_size in enumerate(hidden_dims):
        name = '{}{}'.format(base_layer_name, k)
        '''If this is the 1st hidden layer, then input as input_w_pred;
            else, input the previous hidden_layer'''
        input_now = input_layer if k is 0 else hidden_layer

        hidden_layer = Layer(layer_size, activation = activation, name = name)
        hidden_layer = hidden_layer(input_now)
    
    return hidden_layer

def build_hidden_conv_layers(filter_sizes, kernel_sizes, input_layer, strides,
                        base_layer_name, activation, Layer = Conv1D):
    '''Need to remove all leading zeros for the Decoder 
    to be properly established'''

    filter_sizes = list(filter_sizes)
    while 0 in filter_sizes: filter_sizes.remove(0)
    
    # Establish hidden layer structure
    for k, (filter_size, kernel_size) in enumerate(filter_sizes, kernel_sizes):
        name = '{}{}'.format(base_layer_name, k)
        
        '''If this is the 1st hidden layer, then input as input_layer;
            else, input the previous hidden_layer'''
        input_now = input_layer if k is 0 else hidden_layer

        hidden_layer = Layer(filter_size, kernel_size, 
                                activation = activation, name = name)
        hidden_layer = hidden_layer(input_now)
    
    return hidden_layer

def Conv1DTranspose(filters, kernel_size, strides = 1, padding='same', 
						activation = 'relu', name = None):
	
	conv1D = Sequential()
	conv1D.add(UpSampling1D(size=strides))

	# FINDME maybe strides should be == 1 here? Check size ratios after decoder
	conv1D.add(Conv1D(filters=filters, kernel_size=kernel_size, 
							strides=strides, padding=padding,
							activation = activation, name = name))
	return conv1D

class CustomVariationalLayer(keras.layers.Layer):
	def vae_loss(self, x, z_decoded, z_mean, z_log_var, kl_loss_coeff=5e-4):
		x = K.flatten(x)
		z_decoded = K.flatten(z_decoded)
		
		xent_loss = metrics.binary_crossentropy(x, z_decoded)
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -kl_loss_coeff*K.mean(kl_loss,axis=-1)

		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		# We have to implement custom layers by writing a call method
		input_img, z_decoded, z_mean, z_log_var = inputs
		loss = self.vae_loss(input_img, z_decoded, z_mean, z_log_var)
		self.add_loss(loss, inputs = inputs)

		# you don't use this output; but the layer must return something
		return input_img 

class ConvVAE(object):
	def __init__(self, encoder_filters, encoder_kernel_sizes,
						decoder_filters, decoder_kernel_sizes, 
						encoder_strides=2, decoder_strides=2, 
						latent_dim = 2, encoder_top_size = 16,
						final_kernel_size = 3, data_shape = (784,1), 
						batch_size = 16, run_all = False, 
						verbose = False, plot_model = False):

		self.data_shape = data_shape
		self.batch_size = batch_size
		self.latent_dim = latent_dim
		self.verbose = verbose
		
		self.encoder_filters = encoder_filters
		self.encoder_kernel_sizes = encoder_kernel_sizes
		self.encoder_strides = encoder_strides
		
		self.decoder_filters = decoder_filters
		self.decoder_kernel_sizes = decoder_kernel_sizes
		self.decoder_strides = decoder_strides

		self.encoder_top_size = encoder_top_size
		self.final_kernel_size = final_kernel_size

		if run_all:
			self.load_data()
			self.build_model()
			self.compile()
			
			if plot_model: self.plot_model(save_name = parser.plot_name)

			self.train()

	def build_encoder(self, strides=2, padding='same', 
						activation='relu', base_name = 'enc_conv1d_{}'):
		if self.verbose: print('[INFO] Building Encoder')
		
		x = self.input_img
		cntfilt = 0
		zipper = zip(self.encoder_filters, 
					self.encoder_kernel_sizes,
					self.encoder_strides)

		for kb, (cfilter, ksize, stride) in enumerate(zipper):
			name = base_name.format(cntfilt)
			
			x = layers.Conv1D(filters = cfilter, 
								kernel_size = ksize,
								strides = stride, 
								padding = padding,
								activation = activation, 
								name = name)(x)
			cntfilt += 1

		self.shape_before_flattening = K.int_shape(x)
		x = layers.Flatten()(x)
		# x = layers.Flatten()(x)
		x = layers.Dense(units = self.encoder_top_size, 
						activation='relu', name = 'enc_dense_0')(x)

		# The input image ends up being encoded into these two parameters
		self.z_mean = layers.Dense(units = self.latent_dim, 
								name = 'z_mean')(x)
		self.z_log_var = layers.Dense(units = self.latent_dim, 
								name = 'z_log_var')(x)

	def sampling(self, args, mean=0.0, stddev=1.0):
		if self.verbose: print('[INFO] Sampling Latent Layer')
		z_mean, z_log_var = args
		random_shape = (K.shape(z_mean)[0], self.latent_dim)
		epsilon = K.random_normal(shape=random_shape,
								mean=mean, stddev=stddev)
		return z_mean + K.exp(z_log_var) * epsilon

	def build_decoder(self, strides=2, padding='same', 
					activation='relu', base_name = 'dec_conv1dT_{}'):

		if self.verbose: print('[INFO] Building Decoder')
		# Input where you'll feed z
		z_shape = K.int_shape(self.z)[1:]
		decoder_input = layers.Input(shape = z_shape, name = 'dec_input')

		# Upsamples the input
		x = layers.Dense(np.prod(self.shape_before_flattening[1:]),
							activation = 'relu',
							name = 'dec_dense_0')(decoder_input)

		# Reshapes z into a feature map of the same shape as the feature map
		#	just before the last Flatten layer in the encoder model
		x = layers.Reshape(self.shape_before_flattening[1:])(x)

		''' Uses a Conv1DTranspose layer and a Conv1D layer to decode z into
				a feature map that is the same size as the original image input
		'''
		cntfilt = 0
		
		zipper = zip(self.decoder_filters, 
					self.decoder_kernel_sizes,
					self.decoder_strides)

		for kb, (cfilter, ksize, stride) in enumerate(zipper):
			name = base_name.format(cntfilt)
			x = Conv1DTranspose(filters = cfilter, 
								kernel_size = ksize,
								strides = stride,
								padding = padding, 
								activation = activation,
								name = name)(x)
			
			cntfilt += 1

		# Use a point-wise convolution on the top of the conv-stack
		#	this is the image generation stage: sigmoid == images from 0-1
		one = 1 # required to make a `point-wise` convolution

		# Needed to double-UpSampling because there is only 1 decoder layer
		x = UpSampling1D(size=strides)(x)
		x = UpSampling1D(size=strides)(x) 
		x = Conv1DTranspose(filters =one, kernel_size =self.final_kernel_size, 
							padding = 'same', activation = 'sigmoid',
							strides = strides, name = 'dec_conv1D_pw0')(x)
		
		# Instantiates the decoder model, which turns "decoder_input" into 
		#	the decoded image
		self.decoder_model = Model(decoder_input, x, name='decoder_model')

	def build_model(self):
		if self.verbose: print('[INFO] Building Model')
		self.input_img = layers.Input(shape = self.data_shape)

		# Encodes the input into a mean and variance parameter
		self.build_encoder()

		# Draws a latent point using a small random epsilon
		self.z = layers.Lambda(self.sampling, name='latent_sampling')
		self.z = self.z([self.z_mean, self.z_log_var])

		'''Decode the z back to an image'''
		
		# This function applies the decoding to z 
		#	in order to recover the `z_decoded`
		self.build_decoder()
		self.z_decoded = self.decoder_model(self.z)

		# Calls the custom layer on the input and the decoded output to obtain
		#	the final model output
		self.reconstructed_img = CustomVariationalLayer(
									name = 'CustomVariationalLayer')
		
		self.reconstructed_img = self.reconstructed_img([self.input_img, 
														self.z_decoded,
														self.z_mean, 
														self.z_log_var])

		# Instantiates the autoencoder model, which maps an input image 
		#	to its reconstruction
		self.vae = Model(self.input_img, self.reconstructed_img)

	def load_data(self, divisor = 255.):
		if self.verbose: print('[INFO] Loading Data')
		(x_train, _), (x_test, _) = mnist.load_data()
		x_train = x_train.astype('float32') / divisor
		x_test = x_test.astype('float32') / divisor
		
		train_shape = x_train.shape
		test_shape = x_test.shape 
		x_train = x_train.reshape((train_shape[0], np.prod(train_shape[1:])))
		x_test = x_test.reshape((test_shape[0], np.prod(test_shape[1:])))

		self.x_train = np.expand_dims(x_train, axis=2)
		self.x_test = np.expand_dims(x_test, axis=2)

		if self.verbose: 
			print('[INFO] X_train.shape: {}'.format(self.x_train.shape))
			print('[INFO] X_test.shape: {}'.format(self.x_test.shape))
	
	def compile(self):
		if self.verbose: print('[INFO] Compiling Model')
		# Because the CustomVariationalLayer includes the loss term
		#	we do not need to include a loss term here
		self.vae.compile(optimizer='adam', loss=None)
		if self.verbose: self.vae.summary()

	def train(self):
		if self.verbose: print('[INFO] Training Model')
		self.vae.fit(x = self.x_train, y = None,
				shuffle = True,
				epochs = 10,
				batch_size = self.batch_size,
				validation_data = (self.x_test, None))

	def plot_model(self, save_name): plot_model(self.vae, to_file=save_name)

	def plot_latent_representations(self, n_digits = 15, digit_size = 28, 
									figsize = (10,10), cmap = 'Greys_r'):
		if self.verbose: print('[INFO] Plotting Latent Space Representations')

		''' Sampling the Latent Space '''
		from matplotlib import pyplot as plt
		from scipy.stats import norm

		# This will display a grid of 15x15 digits (255 digits total)
		figure = np.zeros((digit_size * n_digits, digit_size * n_digits))

		''' Transform the linearly spaced coordinates using the SciPy ppf function
				to produce values of the latent variable z (because the prior of the 
				latent space is Gaussian)
		'''
		grid_x = norm.ppf(np.linspace(0.05, 0.95, n_digits))
		grid_y = norm.ppf(np.linspace(0.05, 0.95, n_digits))

		for i, yi in enumerate(grid_x): # why not `grid_y`?
			for j, xi in enumerate(grid_y): # why not `grid_x`?
				z_sample = np.array([[xi,yi]])

				# Repeat z multiple times to form a complete batch
				z_sample = np.tile(z_sample, self.batch_size)
				z_sample = z_sample.reshape(self.batch_size, self.latent_dim)

				# Decode the batch into digit images
				x_decoded = decoder.predict(z_sample, 
								batch_size=self.batch_size)
				
				# Reshapes the first digit in the batch from 28x28x1 to 28x28
				digit = x_decoded[0].reshape(digit_size, digit_size)
				figure[i * digit_size: (i+1) * digit_size,
					   j * digit_size: (j+1) * digit_size] = digit

		plt.figure(figsize=figsize)
		plt.imshow(figure, cmap=cmap)
		plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_all', action='store_true', 
		help='Boolean for starting training now!')
	parser.add_argument('--latent_dim', type=int, default=2, 
		help='Size of the latent layer')
	parser.add_argument('--batch_size', type=int, default=16, 
		help='Size of the batch for SGD')
	parser.add_argument('--verbose', action='store_true', 
		help='Toggle whether to print more statements.')
	parser.add_argument('--encoder_filter_size', type=int, default=16,
		help='Number of filters in the encoder')
	parser.add_argument('--encoder_kernel_size', type=int, default=3,
		help='Kernel size across the encoder')
	parser.add_argument('--encoder_stride', type=int, default=2,
		help='Stride size across the encoder')
	parser.add_argument('--num_encoder_layers', type=int, default=2,
		help='Number of layers in the encoder')
	parser.add_argument('--encoder_top_size', type=int, default=16,
		help='Size of Dense layer on top of encoder')
	parser.add_argument('--decoder_filter_size', type=int, default=32,
		help='Number of filters in the decoder')
	parser.add_argument('--decoder_kernel_size', type=int, default=3,
		help='Kernel size across the decoder')
	parser.add_argument('--decoder_stride', type=int, default=2,
		help='Stride size across the decoder')
	parser.add_argument('--num_decoder_layers', type=int, default=2,
		help='Number of layers in the decoder')
	parser.add_argument('--decoder_bottom_size', type=int, default=16,
		help='Size of Dense layer on top of decoder')
	parser.add_argument('--plot_model', action="store_true", 
		help='Toggle whether to plot the model using keras.utils.plot_model')
	parser.add_argument('--plot_name', type=str, 
		default='Conv1D_VAE_Model_Diagram.png',
		help='Image file name to save model diagram')
	
	parser = parser.parse_args()

	data_shape = (784,1) # MNIST
	batch_size = parser.batch_size
	latent_dim = parser.latent_dim
	verbose = parser.verbose
	run_all = parser.run_all

	''' Configure encoder '''
	n_encoder_layers = parser.num_encoder_layers
	
	encoder_filters = np.array([parser.encoder_filter_size]*n_encoder_layers)
	encoder_filters = encoder_filters*(2**np.arange(n_encoder_layers))
	
	encoder_kernel_sizes = [parser.encoder_kernel_size]*n_encoder_layers
	encoder_strides = [parser.encoder_stride]*n_encoder_layers

	''' Configure Decoder '''
	n_decoder_layers = parser.num_decoder_layers
	
	decoder_filters = np.array([parser.decoder_filter_size]*n_decoder_layers)
	decoder_filters = decoder_filters//(2**np.arange(n_decoder_layers))

	decoder_kernel_sizes = [parser.decoder_kernel_size]*n_decoder_layers
	decoder_strides = [parser.decoder_stride]*n_decoder_layers

	vae = ConvVAE(encoder_filters = encoder_filters, 
			encoder_kernel_sizes = encoder_kernel_sizes,
			decoder_filters = decoder_filters, 
			decoder_kernel_sizes = decoder_kernel_sizes,
			encoder_strides=encoder_strides,
			decoder_strides=decoder_strides,
			latent_dim = latent_dim, 
			encoder_top_size = parser.encoder_top_size,
			final_kernel_size = parser.decoder_kernel_size,
			data_shape = data_shape, 
			batch_size = batch_size,
			verbose = verbose, 
			run_all = run_all, 
			plot_model = parser.plot_model)

	if not run_all:
		vae.load_data()
		vae.build_model()
		vae.compile()
