import json
import numpy as np
import scipy.stats

from functools import partial, update_wrapper

from keras import backend as K
from keras.layers import Input, Lambda, concatenate, Dense, Conv1D,UpSampling1D
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.losses import mean_squared_error

ITERABLES = (list, tuple, np.array)

# try:
# 	# Python 2 
# 	range 
# except: 
# 	# Python 3
# 	def range(tmp): return iter(range(tmp))

'''HERE WHERE I STARTED'''
def Conv1DTranspose(filters, kernel_size, strides = 1, padding='same', 
						activation = 'relu', name = None):
	
	conv1D = Sequential()
	conv1D.add(UpSampling1D(size=strides))

	# FINDME maybe strides should be == 1 here? Check size ratios after decoder
	conv1D.add(Conv1D(filters=filters, kernel_size=(kernel_size), 
							strides=strides, padding=padding,
							activation = activation, name = name))
	return conv1D

def vae_sampling(args, latent_dim, batch_size = 128, 
					mean = 0.0, stddev = 1.0, channels = None):
	latent_mean, latent_log_var = args
	
	if channels is not None:
		batch_shape = (batch_size, channels, latent_dim)
	else:
		batch_shape = (batch_size, latent_dim)

	eps = K.random_normal(shape = batch_shape, mean = mean, stddev = stddev)
	
	return latent_mean + K.exp(latent_log_var/2) * eps

def dnn_sampling(args, latent_dim, batch_size = 128, channels = None,
				mean = 0.0, stddev = 1.0,  name = 'dnn_norm'):
	
	latent_mean, latent_log_var = args
	
	if channels is not None:
		batch_shape = (batch_size, channels, latent_dim)
	else:
		batch_shape = (batch_size, latent_dim)
	
	eps = K.random_normal(shape = batch_shape, mean = mean, stddev = stddev)

	gamma_ = K.exp(latent_log_var/2)*eps
	mean_norm = latent_mean + gamma_
	
	# need to add 0's so we can sum it all to 1
	padding = K.tf.zeros(batch_size, 1)[:,None]
	# mean_norm = concatenate([mean_norm, padding], name=name)
	sum_exp_dnn_norm = K.sum(K.exp(mean_norm), axis = -1)[:,None]
	
	return K.exp(mean_norm)/sum_exp_dnn_norm

def build_hidden_layers(hidden_dims, input_layer, Layer = None, 
				kernel_sizes = None, strides=None, base_layer_name = '', 
				activation = 'relu'):
	
	if 'conv' in Layer.lower() and kernel_sizes is not None:
		assert(len(kernel_sizes) == len(hidden_dims)), \
			"each Conv layer requires a given kernel_size"
		Layer = Conv1D
		return build_hidden_conv_layers(filter_sizes = hidden_dims, 
										input_layer = input_layer, 
										kernel_sizes = kernel_sizes, 
										strides = strides, 
										base_layer_name = base_layer_name, 
										activation = activation, 
										Layer = Layer)

def build_hidden_conv_layers(filter_sizes, input_layer, Layer = 'conv1d', 
							kernel_sizes = 3, strides = 2, 
							base_layer_name = '', activation  = 'relu'):
	'''Need to remove all leading zeros for the Decoder 
		to be properly established'''
	while 0 in filter_sizes: filter_sizes.remove(0)
	
	# if isinstance(Layer,str): Layer = layer_selection[Layer.lower()]
	Layer = Conv1D	
	# Establish hidden layer structure
	zipper = zip(filter_sizes, kernel_sizes, strides)
	for k, (filter_size, kernel_size, stride) in enumerate(zipper):
		name = '{}{}'.format(base_layer_name, k)
		
		'''If this is the 1st hidden layer, then input as input_layer;
			else, input the previous hidden_layer'''
		input_now = input_layer if k is 0 else hidden_layer
		
		print('input_now', input_now)
		print('filter_size',filter_size) 
		print('kernel_size',kernel_size) 
		print('stride', stride)
		print('Layer', Layer(filter_size, (kernel_size), strides=stride,
						activation = activation, name = name))
		hidden_layer = Layer(filter_size, (kernel_size), strides=stride,
						activation = activation, name = name)(input_now)

		print('hidden_layer',hidden_layer)

	return hidden_layer

class ConVAEPredictor(object):
	def __init__(self, encoder_filters, encoder_kernel_sizes,
					decoder_filters, decoder_kernel_sizes, 
					dnn_filters, dnn_kernel_sizes, dnn_strides = 2,
					encoder_strides = 2, decoder_strides = 2, 
					vae_latent_dim = 2, encoder_top_size = 16, 
					final_kernel_size = 3, data_shape = (784,1), 
					batch_size = 128, run_all = False, 
					verbose = False, plot_model = False,
					original_dim = None, dnn_out_dim = None, 
					dnn_latent_dim = None, batch_size = 128, 
					dnn_log_var_prior = 0.0, optimizer = 'adam-wn', 
					predictor_type = 'classification', layer_type = 'Conv1D'):

		self.data_shape = data_shape
		self.batch_size = batch_size
		self.vae_latent_dim = vae_latent_dim
		self.verbose = verbose
		
		self.encoder_filters = encoder_filters
		self.encoder_kernel_sizes = encoder_kernel_sizes
		self.encoder_strides = encoder_strides
		self.encoder_top_size = encoder_top_size
		
		self.decoder_filters = decoder_filters
		self.decoder_kernel_sizes = decoder_kernel_sizes
		self.decoder_strides = decoder_strides
		self.final_kernel_size = final_kernel_size

		self.dnn_filters = dnn_filters
		self.dnn_kernel_sizes = dnn_kernel_sizes
		self.dnn_strides = dnn_strides
		self.dnn_out_dim = dnn_out_dim

		self.layer_type = layer_type
		self.predictor_type = predictor_type
		self.original_dim = original_dim
		
		"""FINDME: Why is this dnn_out_dim-1(??)"""
		if dnn_latent_dim is not None: self.dnn_latent_dim = dnn_out_dim - 1

		self.dnn_log_var_prior = dnn_log_var_prior
		self.optimizer = optimizer
		self.batch_size = batch_size
		# self.use_prev_input = use_prev_input
	
	def build_predictor(self, padding='same', activation='relu', 
								base_name = 'dnn_conv1d_{}'):

		if self.verbose: print('[INFO] Building Predictor')
		
		x = self.input_layer

		zipper = zip(self.dnn_filters, 
					self.dnn_kernel_sizes,
					self.dnn_strides)

		for kb, (cfilter, ksize, stride) in enumerate(zipper):
			name = base_name.format(kb)
			
			x = layers.Conv1D(filters = cfilter, 
								kernel_size = ksize,
								strides = stride, 
								padding = padding,
								activation = activation, 
								name = name)(x)

		x = layers.Flatten()(x)
		x = layers.Dense(units = self.dnn_out_dim, 
						activation='relu', name = 'dnn_dense_0')(x)

		# The input image ends up being encoded into these two parameters
		self.dnn_latent_mean = layers.Dense(units = self.dnn_latent_dim, 
								name = 'dnn_latent_mean')(x)
		self.dnn_latent_log_var = layers.Dense(units = self.dnn_latent_dim, 
								name = 'dnn_latent_log_var')(x)

		# Draws a latent point using a small random epsilon
		self.dnn_latent = layers.Lambda(
							self.dnn_sampling, name = 'dnn_latent_sampling')

		self.dnn_latent = self.dnn_latent(
							[self.dnn_latent_mean, self.dnn_latent_log_var])

	def orig_build_predictor(self):
		if bool(sum(self.dnn_filter_sizes)):
			''' Establish dnn Encoder layer structure '''
			dnn_enc_hid_layer = build_hidden_conv_layers(
									filter_sizes = self.dnn_filter_sizes,
									kernel_sizes = self.dnn_kernel_sizes,
									input_layer = self.input_layer, 
									strides = self.dnn_strides,
									base_layer_name = 'dnn_enc_hidden_layer',
									activation  = self.hidden_activation,
									Layer = self.layer_type)
		else:
			'''if there are no hidden layers, then the input to the 
				dnn latent layers is the input_w_pred layer'''
			dnn_enc_hid_layer = self.input_w_pred
		
		self.dnn_latent_mean = Dense(self.dnn_latent_dim,
									name='dnn_latent_mean')(dnn_enc_hid_layer)
		self.dnn_latent_log_var =Dense(self.dnn_latent_dim,
								name = 'dnn_latent_log_var')(dnn_enc_hid_layer)
		
		dnn_latent_layer = Lambda(dnn_sampling, name='dnn_latent_layer',
								arguments = {'latent_dim':self.dnn_latent_dim, 
											'batch_size':self.batch_size,
											'channels':self.n_channels})

		self.dnn_latent_layer = dnn_latent_layer([self.dnn_latent_mean, 
												  self.dnn_latent_log_var])

	def build_latent_encoder(self, padding='same', activation='relu', 
								base_name = 'enc_conv1d_{}'):
		
		if self.verbose: print('[INFO] Building Encoder')
		
		x = self.input_w_pred

		zipper = zip(self.encoder_filters, 
					self.encoder_kernel_sizes,
					self.encoder_strides)

		for kb, (cfilter, ksize, stride) in enumerate(zipper):
			name = base_name.format(kb)
			
			x = layers.Conv1D(filters = cfilter, 
								kernel_size = ksize,
								strides = stride, 
								padding = padding,
								activation = activation, 
								name = name)(x)

		self.last_conv_shape = K.int_shape(x)

		x = layers.Flatten()(x)
		
		x = layers.Dense(units = self.encoder_top_size, 
						activation='relu', name = 'enc_dense_0')(x)

		# The input image ends up being encoded into these two parameters
		self.vae_latent_mean = layers.Dense(units = self.vae_latent_dim, 
								name = 'vae_latent_mean')(x)
		self.vae_latent_log_var = layers.Dense(units = self.vae_latent_dim, 
								name = 'vae_latent_log_var')(x)

		# Draws a latent point using a small random epsilon
		self.vae_latent = layers.Lambda(self.vae_sampling, 
										name='vae_latent_sampling')
		self.vae_latent = self.vae_latent([self.vae_latent_mean, 
											self.vae_latent_log_var])

		self.vae_latent_args = concatenate([self.vae_latent_mean, 
											self.vae_latent_log_var], 
											axis = -1, 
											name = 'vae_latent_args')

	def vae_sampling(self, args, mean=0.0, stddev=1.0):
		if self.verbose: print('[INFO] Sampling Latent Layer')
		z_mean, z_log_var = args
		random_shape = (K.shape(z_mean)[0], self.vae_latent_dim)
		epsilon = K.random_normal(shape=random_shape,
								mean=mean, stddev=stddev)
		return z_mean + K.exp(z_log_var) * epsilon

	def orig_build_latent_encoder(self):
		if bool(sum(self.vae_hidden_dims)):
			''' Establish VAE Encoder layer structure '''
			vae_enc_hid_layer = build_hidden_conv_layers(
									filter_sizes = self.vae_enc_filters_sizes,
									kernel_sizes = self.vae_kernel_sizes,
									input_layer = self.input_w_pred,
									strides = self.vae_strides,
									base_layer_name = 'vae_enc_hidden_layer',
									activation  = self.hidden_activation,
									Layer = self.layer_type)
		else:
			'''if there are no hidden layers, then the input to the 
				vae latent layers is the input_w_pred layer'''
			vae_enc_hid_layer = self.input_w_pred
		
		self.vae_encoder_output_shape = K.int_shape(vae_enc_hid_layer)

		self.vae_latent_mean = Dense(self.vae_latent_dim,
									name='vae_latent_mean')(vae_enc_hid_layer)
		self.vae_latent_log_var =Dense(self.vae_latent_dim,
								name = 'vae_latent_log_var')(vae_enc_hid_layer)
		
		vae_latent_layer = Lambda(vae_sampling, name='vae_latent_layer',
								arguments = {'latent_dim':self.vae_latent_dim, 
											'batch_size':self.batch_size,
											'channels':self.n_channels})

		self.vae_latent_layer = vae_latent_layer([self.vae_latent_mean, 
												  self.vae_latent_log_var])

		self.vae_latent_args = concatenate([self.vae_latent_mean, 
													self.vae_latent_log_var], 
													axis = -1, 
													name = 'vae_latent_args')
	
	def build_latent_decoder(self, padding='same', activation='relu', 
						base_name = 'dec_conv1dT_{}'):

		if self.verbose: print('[INFO] Building Decoder')
		# Input where you'll feed z
		z_shape = K.int_shape(self.z)[1:]
		decoder_input = layers.Input(shape = z_shape, name = 'dec_input')
		
		# Upsamples the input
		x = layers.Dense(np.prod(self.last_conv_shape[1:]),
							activation = 'relu',
							name = 'dec_dense_0')(decoder_input)

		# Reshapes z into a feature map of the same shape as the feature map
		#	just before the last Flatten layer in the encoder model
		x = layers.Reshape(self.last_conv_shape[1:])(x)
		
		''' Uses a Conv1DTranspose layer and a Conv1D layer to decode z into
				a feature map that is the same size as the original image input
		'''
		zipper = zip(self.decoder_filters, 
					self.decoder_kernel_sizes,
					self.decoder_strides)

		for kb, (cfilter, ksize, stride) in enumerate(zipper):
			name = base_name.format(kb)
			
			x = Conv1DTranspose(filters = cfilter, 
								kernel_size = ksize,
								strides = stride,
								padding = padding, 
								activation = activation,
								name = name)(x)
		
		# Use a point-wise convolution on the top of the conv-stack
		#	this is the image generation stage: sigmoid == images from 0-1
		one = 1 # required to make a `point-wise` convolution
		x = Conv1DTranspose(filters =one, kernel_size =self.final_kernel_size, 
							padding = 'same', activation = 'sigmoid',
							name = 'dec_conv1D_pw0')(x) # strides = one, 
		
		# Instantiates the decoder model, which turns "decoder_input" into 
		#	the decoded image
		self.decoder_model = Model(decoder_input, x, name='decoder_model')
		model_shape = K.int_shape(self.decoder_model(decoder_input))

	def orig_build_latent_decoder(self):
		
		if len(self.vae_encoder_output_shape) > 2:
			# Upscale and resize to prepare for deconvolution
			enc_size_ = np.prod(self.vae_encoder_output_shape[1:])
			upscale = Dense(enc_size_, activation = self.hidden_activation, 
							name = 'upscale_layer')
			upscale = upscale(self.dnn_w_latent)
			reshape_ = Reshape(enc_shape[1:], name = 'reshape')(upscale)
		else:
			reshape_ = self.dnn_w_latent

		if bool(sum(self.vae_hidden_dims)):
			# reverse order for decoder
			vae_dec_hid_layer = build_hidden_conv_layers(
									filter_sizes = self.vae_hidden_filter_size,
									num_filters = self.num_vae_hidden_layers,
									kernel_sizes = self.vae_hidden_kernel_size,
									input_layer = reshape_, # reshaped input
									strides = self.strides,
									base_layer_name = 'vae_dec_hidden_layer',
									activation  = self.hidden_activation,
									Layer = '{}T'.format(self.layer_type))
		else:
			vae_dec_hid_layer = self.dnn_w_latent
		
		one = 1
		vae_reconstruction = Conv1DTranspose(filters = one,
								  kernel_size = self.vae_hidden_kernel_size,
								  activation = self.output_activation,
								  padding = 'same',
								  name = 'vae_reconstruction')
		
		self.vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)

	def dnn_kl_loss(self, labels, preds):
		vs = 1 - self.dnn_log_var_prior + self.dnn_latent_log_var
		vs = vs - K.exp(self.dnn_latent_log_var)/K.exp(self.dnn_log_var_prior)
		vs = vs - K.square(self.dnn_latent_mean)/K.exp(self.dnn_log_var_prior)

		return -0.5*K.sum(vs, axis = -1)

	def dnn_rec_loss(self, labels, preds):
		
		if self.predictor_type is 'classification':
			rec_loss = categorical_crossentropy(labels, preds)
			# rec_loss = self.dnn_latent_dim * rec_loss
		elif self.predictor_type is 'regression':
			rec_loss = mean_squared_error(labels, preds)
		else:
			rec_loss = categorical_crossentropy(labels, preds)
		
		return rec_loss

	# def dnn_sampling(self, args):
	# 	''' sample from a logit-normal with params dnn_latent_mean 
	# 			and dnn_latent_log_var
	# 		(n.b. this is very similar to a logistic-normal distribution)
	# 	'''
	# 	batch_shape = (self.batch_size, self.dnn_latent_dim)
	# 	eps = K.random_normal(shape = batch_shape, mean = 0., stddev = 1.0)

	# 	gamma_ = K.exp(self.dnn_latent_log_var/2)*eps
	# 	dnn_norm = self.dnn_latent_mean + gamma_
		
	# 	# need to add 0's so we can sum it all to 1
	# 	padding = K.tf.zeros(self.batch_size, 1)[:,None]
	# 	dnn_norm = concatenate([dnn_norm, padding], 
	# 							name='dnn_norm')
	# 	sum_exp_dnn_norm = K.sum(K.exp(dnn_norm), axis = -1)[:,None]
	# 	return K.exp(dnn_norm)/sum_exp_dnn_norm

	def get_model(self, num_gpus = 0, batch_size = None, original_dim = None, 
				  vae_hidden_dims = None, vae_latent_dim = None, 
				  dnn_hidden_dims = None, # use_prev_input = False, 
				  dnn_log_var_prior = 0.0, hidden_activation = 'relu', 
				  output_activation = 'sigmoid'):

		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		
		if dnn_log_var_prior is not None:
			self.dnn_log_var_prior = dnn_log_var_prior
		
		# Update new input values from local args
		self.batch_size = batch_size or self.batch_size
		self.original_dim = original_dim or self.original_dim
		self.vae_hidden_dims = vae_hidden_dims or self.vae_hidden_dims
		self.vae_latent_dim = vae_latent_dim or self.vae_latent_dim
		self.dnn_hidden_dims = dnn_hidden_dims or self.dnn_hidden_dims
		
		# if use_prev_input is not None: self.use_prev_input = use_prev_input

		batch_shape = (self.batch_size, self.original_dim, self.n_channels)
		# np.expand_dims(X, axis=2)
		# batch_shape = (self.original_dim,)
		self.input_layer = Input(batch_shape = batch_shape, name='input_layer')

		# if use_prev_input or self.use_prev_input:
		# 	self.prev_input_layer = Input(batch_shape = batch_shape, 
		# 								name = 'previous_input_layer')
		self.build_predictor()
		
		self.input_w_pred = concatenate([self.input_layer, 
										self.dnn_latent_layer], 
										axis = -1,
										name = 'data_input_w_dnn_latent_out')

		self.build_latent_encoder()
		
		# if use_prev_input or self.use_prev_input:
		# 	self.prev_w_vae_latent = concatenate(
		# 		[self.prev_input_layer, self.vae_latent_layer], 
		# 		 axis = -1, name = 'prev_inp_w_vae_lat_layer')
		# else:
		# 	self.prev_w_vae_latent = self.vae_latent_layer
		
		self.dnn_w_latent = concatenate(
						[self.dnn_latent_layer, self.vae_latent_layer],
						axis = -1, name = 'dnn_latent_out_w_prev_w_vae_lat')
		
		self.build_latent_decoder()
		
		# if use_prev_input or self.use_prev_input:
		# 	input_stack = [self.input_layer, self.prev_input_layer]
		# 	out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
		# 				 self.dnn_latent_mod, self.vae_latent_args]
		# 	enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]
		# else:
		# input_stack = self.input_layer
		out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
					 self.dnn_latent_mod, self.vae_latent_args]
		# enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]

		self.model = Model(self.input_layer, out_stack)
		# self.enc_model = Model(self.input_layer, enc_stack)

		#Make The Model Parallel Using Multiple GPUs
		#if(num_gpus >= 2):
		#	self.model = multi_gpu_model(self.model, gpus=num_gpus)

	def compile(self, dnn_weight = 1.0, vae_weight = 1.0, vae_kl_weight = 1.0, 
				  dnn_kl_weight = 1.0, optimizer = None, metrics = None):
		
		self.model.compile(
				optimizer = optimizer or self.optimizer,

				loss = {'vae_reconstruction': self.vae_loss,
						'dnn_latent_layer': self.dnn_kl_loss,
						'predictor_latent_mod': self.dnn_rec_loss,
						'vae_latent_args': self.vae_kl_loss},

				loss_weights = {'vae_reconstruction': vae_weight,
								'dnn_latent_layer': dnn_kl_weight,
								'predictor_latent_mod':dnn_weight,
								'vae_latent_args': vae_kl_weight},

				metrics = metrics or {'dnn_latent_layer': ['acc', 'mse']})
	
	def vae_loss(self, input_layer, vae_reconstruction):
		if self.predictor_type is 'binary':
			'''This case is specific to binary input sequences
				i.e. [0,1,1,0,0,0,....,0,1,1,1]'''
			inp_vae_loss = binary_crossentropy(input_layer, vae_reconstruction)
			inp_vae_loss = self.original_dim * inp_vae_loss
		
		elif self.predictor_type is 'classification':
			''' I am left to assume that the vae_reconstruction_loss for a 
					non-binary data source (i.e. *not* piano keys) should be 
					a regression problem. 
				The prediction loss is still categorical crossentropy because 
					the features being compared are discrete classes'''
			inp_vae_loss = mean_squared_error(input_layer, vae_reconstruction)
		
		elif self.predictor_type is 'regression':
			inp_vae_loss = mean_squared_error(input_layer, vae_reconstruction)
		else:
			inp_vae_loss = mean_squared_error(input_layer, vae_reconstruction)
		
		return inp_vae_loss

	def vae_kl_loss(self, ztrue, zpred):
		Z_mean = self.vae_latent_args[:,:self.vae_latent_dim]
		Z_log_var = self.vae_latent_args[:,self.vae_latent_dim:]
		k_summer = 1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var)
		return -0.5*K.sum(k_summer, axis = -1)

	def load_model(self, model_file):
		''' there's a currently bug in the way keras loads models 
				from `.yaml` files that has to do with `Lambda` calls
				... so this is a hack for now
		'''
		self.get_model()
		self.model.load_weights(model_file)

	def make_dnn_predictor(self):
		batch_shape = (self.batch_size, self.original_dim, self.n_channels)
		# batch_shape = (self.original_dim,)
		input_layer = Input(batch_shape = batch_shape, 
									name = 'input_layer')

		# build label encoder
		enc_hidden_layer = self.model.get_layer('predictor_hidden_layer')
		enc_hidden_layer = enc_hidden_layer(input_layer)
		
		dnn_latent_mean = self.model.get_layer('predictor_latent_mean')
		dnn_latent_mean = dnn_latent_mean(enc_hidden_layer)

		dnn_latent_log_var = self.model.get_layer('predictor_latent_log_var')
		dnn_latent_log_var = dnn_latent_log_var(enc_hidden_layer)

		return Model(input_layer, [dnn_latent_mean, dnn_latent_log_var])

	def make_latent_encoder(self):
		orig_batch_shape = (self.batch_size, self.original_dim,self.n_channels)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		# orig_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.predictor_out_dim,)

		input_layer = Input(batch_shape = orig_batch_shape, 
							name = 'input_layer')
		dnn_input_layer = Input(batch_shape = predictor_batch_shape, 
							name = 'predictor_input_layer')

		input_w_pred = concatenate([input_layer, dnn_input_layer], 
									axis = -1, name = 'input_w_dnn_layer')
		
		# build latent encoder
		if self.vae_hidden_dim > 0:
			vae_enc_hid_layer = self.model.get_layer('vae_enc_hid_layer')
			vae_enc_hid_layer = vae_enc_hid_layer(input_w_pred)

			vae_latent_mean = self.model.get_layer('vae_latent_mean')
			vae_latent_mean = vae_latent_mean(vae_enc_hid_layer)

			vae_latent_log_var = self.model.get_layer('vae_latent_log_var')
			vae_latent_log_var = vae_latent_log_var(vae_enc_hid_layer)
		else:
			vae_latent_mean = self.model.get_layer('vae_latent_mean')
			vae_latent_mean = vae_latent_mean(input_w_pred)
			vae_latent_log_var= self.model.get_layer('vae_latent_log_var')
			vae_latent_log_var = vae_latent_log_var(input_w_pred)
		
		latent_enc_input = [input_layer, dnn_input_layer]
		latent_enc_output = [vae_latent_mean, vae_latent_log_var]

		return Model(latent_enc_input, latent_enc_output)

	def make_latent_decoder(self):# , use_prev_input=False

		input_batch_shape = (self.batch_size,self.original_dim,self.n_channels)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		vae_batch_shape = (self.batch_size,self.vae_latent_dim,self.n_channels)
		# input_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.dnn_out_dim,)
		# vae_batch_shape = (self.vae_latent_dim,)

		dnn_input_layer = Input(batch_shape = predictor_batch_shape, 
										name = 'predictor_layer')
		vae_latent_layer = Input(batch_shape = vae_batch_shape, 
									name = 'vae_latent_layer')

		# if use_prev_input or self.use_prev_input:
		# 	prev_input_layer = Input(batch_shape = input_batch_shape, 
		# 								name = 'prev_input_layer')

		# if use_prev_input or self.use_prev_input:
		# 	prev_vae_stack = [prev_input_layer, vae_latent_layer]
		# 	prev_w_vae_latent = concatenate(prev_vae_stack, axis = -1)
		# else:
		# prev_w_vae_latent = vae_latent_layer

		dnn_w_latent = concatenate([dnn_input_layer, vae_latent_layer],
										axis = -1)

		# build physical decoder
		vae_reconstruction = self.model.get_layer('vae_reconstruction')
		if self.vae_hidden_dim > 0:
			vae_dec_hid_layer = self.model.get_layer('vae_dec_hid_layer')
			vae_dec_hid_layer = vae_dec_hid_layer(dnn_w_latent)
			vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)
		else:
			vae_reconstruction = vae_reconstruction(dnn_w_latent)

		# if use_prev_input or self.use_prev_input:
		# 	dec_input_stack = [dnn_input_layer, 
		# 						vae_latent_layer, 
		# 						prev_input_layer]
		# else:
		dec_input_stack = [dnn_input_layer, vae_latent_layer]

		return Model(dec_input_stack, vae_reconstruction)

	def sample_predictor(self, dnn_latent_mean, 
								dnn_latent_log_var, 
								nsamps=1, nrm_samp=False, 
								add_noise=True):
		
		if nsamps == 1:
			eps_shape = dnn_latent_mean.flatten().shape[0]
			eps = np.random.randn(*((1, eps_shape)))
		else:
			eps = np.random.randn(*((nsamps,) + dnn_latent_mean.shape))
		if eps.T.shape == dnn_latent_mean.shape:
			eps = eps.T

		gamma_ = np.exp(dnn_latent_log_var/2)
		if add_noise:
			dnn_norm = dnn_latent_mean + gamma_*eps
		else:
			dnn_norm = dnn_latent_mean + 0*gamma_*eps

		if nrm_samp: return dnn_norm

		if nsamps == 1:
			padding = np.zeros((dnn_norm.shape[0], 1))
			dnn_norm = np.hstack([dnn_norm, padding])
			out = np.exp(dnn_norm)/np.sum(np.exp(dnn_norm),axis=-1)
			return out[:,None]
		else:
			padding = np.zeros(dnn_norm.shape[:-1]+(1,))
			dnn_norm = np.dstack([dnn_norm,padding])
			out = np.exp(dnn_norm)/np.sum(np.exp(dnn_norm),axis=-1)
			return out[:,:,None]

	def sample_latent(self, Z_mean, Z_log_var, nsamps = 1):
		if nsamps == 1:
			eps = np.random.randn(*Z_mean.squeeze().shape)
		else:
			eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
		return Z_mean + np.exp(Z_log_var/2) * eps

	def sample_vae(self, dnn_latent_mean):
		rando = np.random.rand(len(dnn_latent_mean.squeeze()))

		return np.float32(rando <= dnn_latent_mean)

	def save(self):
		joblib_save_loc = '{}/{}_{}_{}_trained_model_output_{}.joblib.save'
		joblib_save_loc = joblib_save_loc.format(self.model_dir, self.run_name,
										 self.generationID, self.chromosomeID,
										 self.time_stamp)

		wghts_save_loc = '{}/{}_{}_{}_trained_model_weights_{}.save'
		wghts_save_loc = wghts_save_loc.format(self.model_dir, self.run_name,
										 self.generationID, self.chromosomeID,
										 self.time_stamp)
		
		model_save_loc = '{}/{}_{}_{}_trained_model_full_{}.save'
		model_save_loc = model_save_loc.format(self.model_dir, self.run_name,
										 self.generationID, self.chromosomeID,
										 self.time_stamp)
		
		self.neural_net.save_weights(wghts_save_loc, overwrite=True)
		self.neural_net.save(model_save_loc, overwrite=True)

		try:
			joblib.dump({'best_loss':self.best_loss,'history':self.history}, 
						joblib_save_loc)
		except Exception as e:
			print(str(e))