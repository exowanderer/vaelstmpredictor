import json
import numpy as np
import scipy.stats

from functools import partial, update_wrapper

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.losses import mean_squared_error
#from keras.utils.multi_gpu_utils import multi_gpu_model

# from ..utils.midi_utils import write_sample

try:
	# Python 2 
	range 
except: 
	# Python 3
   def range(tmp): return iter(range(tmp))

'''HERE WHERE I STARTED'''
def wrapped_partial(func, *args, **kwargs):
	''' from: http://louistiao.me/posts/
	adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
	'''
	partial_func = partial(func, *args, **kwargs)
	update_wrapper(partial_func, func)
	return partial_func

def build_hidden_layers(hidden_dims, input_layer, Layer = None, 
				kernel_sizes = None, strides=None, base_layer_name = '', 
				activation = 'relu'):
	
	assert('conv' in Layer.lower() or 'dense' in Layer.lower()),\
			"Only 'conv' and 'dense' layers are supported here."

	if 'dense' in Layer.lower() or kernel_sizes is None:
		if 'dense' == Layer.lower(): Layer = layers.Dense
		return build_hidden_dense_layers(hidden_dims = hidden_dims, 
										input_layer = input_layer, 
										base_layer_name = base_layer_name, 
										activation = activation, 
										Layer = layers.Dense)

	if 'conv' in Layer.lower() and kernel_sizes is not None:
		assert(len(kernel_sizes) == len(hidden_dims)), \
			"each Conv layer requires a given kernel_size"

		if 'conv1d' == Layer.lower(): Layer = layers.Conv1D
		if 'conv2d' == Layer.lower(): Layer = layers.Conv2D
		if 'conv3d' == Layer.lower(): Layer = layers.Conv3D
		if 'conv1dT' == Layer.lower(): Layer = layers.Conv1DTranspose
		if 'conv2dT' == Layer.lower(): Layer = layers.Conv2DTranspose
		if 'conv3dT' == Layer.lower(): Layer = layers.Conv3DTranspose

		return build_hidden_conv_layers(filter_sizes = hidden_dims, 
										input_layer = input_layer, 
										kernel_sizes = kernel_sizes, 
										strides = strides, 
										base_layer_name = base_layer_name, 
										activation = activation, 
										Layer = Layer)

def build_hidden_dense_layers(hidden_dims, input_layer, base_layer_name, 
								activation, Layer = layers.Dense):
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
						base_layer_name, activation, Layer = None):
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

class VAEPredictor(object):
	def __init__(self, original_dim, vae_hidden_dims, dnn_hidden_dims, 
					vae_latent_dim, dnn_out_dim = None, 
					dnn_latent_dim = None, batch_size = 128, 
					dnn_log_var_prior = 0.0, optimizer = 'adam-wn', 
					use_prev_input = False, predictor_type = 'classification'):
		
		self.predictor_type = predictor_type
		self.original_dim = original_dim
		
		self.vae_hidden_dims = vae_hidden_dims
		self.vae_latent_dim = vae_latent_dim

		self.dnn_hidden_dims = dnn_hidden_dims
		self.dnn_out_dim = dnn_out_dim
		
		"""FINDME: Why is this dnn_out_dim-1(??)"""
		if dnn_latent_dim is not None:
			self.dnn_latent_dim = self.dnn_out_dim - 1

		self.dnn_log_var_prior = dnn_log_var_prior
		self.optimizer = optimizer
		self.batch_size = batch_size
		self.use_prev_input = use_prev_input
	
	def build_predictor(self):
		if bool(sum(self.dnn_hidden_dims)):
			''' Establish VAE Encoder layer structure '''
			dnn_hidden_layer = build_hidden_layers(Layer = 'Dense',
									hidden_dims = self.dnn_hidden_dims, 
									input_layer = self.input_layer, 
									base_layer_name='predictor_hidden_layer', 
									activation=self.hidden_activation)
		else:
			'''if there are no hidden layers, then the input to the 
				dnn latent layers is the input_layer'''
			dnn_hidden_layer = self.input_layer
		
		# Process the predictor layer through a latent layer
		dnn_latent_mean = layers.Dense(self.dnn_latent_dim, 
										name = 'predictor_latent_mean')
		self.dnn_latent_mean = dnn_latent_mean(dnn_hidden_layer)
		dnn_latent_log_var = layers.Dense(self.dnn_latent_dim, 
									name = 'predictor_latent_log_var')
		
		self.dnn_latent_log_var = dnn_latent_log_var(
													dnn_hidden_layer)
		
		dnn_latent_layer = layers.Lambda(self.dnn_sampling, 
									name = 'predictor_latent_layer')
		self.dnn_latent_layer = dnn_latent_layer(
							[self.dnn_latent_mean, self.dnn_latent_log_var])
		
		# Add some wiggle to the dnn predictions 
		#   to avoid division by zero
		dnn_latent_mod = layers.Lambda(lambda tmp: tmp+1e-10, 
								name = 'predictor_latent_mod')
		self.dnn_latent_mod = dnn_latent_mod(self.dnn_latent_layer)
		
	def build_latent_decoder(self):
		if bool(sum(self.vae_hidden_dims)):
			# reverse order for decoder
			vae_dec_hid_layer = build_hidden_layers(Layer = 'Dense',
									hidden_dims = self.vae_hidden_dims[::-1],
									input_layer = self.dnn_w_latent,
									base_layer_name = 'vae_dec_hidden_layer', 
									activation = self.hidden_activation)
		else:
			vae_dec_hid_layer = self.dnn_w_latent
		
		vae_reconstruction = layers.Dense(self.original_dim, 
								 activation = self.output_activation, 
								 name = 'vae_reconstruction')
		self.vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)

	def build_latent_encoder(self):
		if bool(sum(self.vae_hidden_dims)):
			''' Establish VAE Encoder layer structure '''
			vae_enc_hid_layer = build_hidden_layers(Layer = 'Dense',
								hidden_dims = self.vae_hidden_dims, 
								input_layer = self.input_w_pred, 
								base_layer_name='vae_enc_hidden_layer', 
								activation=self.hidden_activation)
		else:
			'''if there are no hidden layers, then the input to the 
				vae latent layers is the input_w_pred layer'''
			vae_enc_hid_layer = self.input_w_pred
		
		vae_latent_mean = layers.Dense(self.vae_latent_dim,name='vae_latent_mean')
		self.vae_latent_mean = vae_latent_mean(vae_enc_hid_layer)
		
		vae_latent_log_var =layers.Dense(self.vae_latent_dim,
								  name = 'vae_latent_log_var')
		self.vae_latent_log_var = vae_latent_log_var(vae_enc_hid_layer)
		
		vae_latent_layer = layers.Lambda(self.vae_sampling, name = 'vae_latent_layer')
		self.vae_latent_layer = vae_latent_layer([self.vae_latent_mean, 
												  self.vae_latent_log_var])

		self.vae_latent_args = concatenate([self.vae_latent_mean, 
											self.vae_latent_log_var], 
											axis = -1, 
											name = 'vae_latent_args')
	
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

	def dnn_sampling(self, args):
		''' sample from a logit-normal with params dnn_latent_mean 
				and dnn_latent_log_var
			(n.b. this is very similar to a logistic-normal distribution)
		'''
		batch_shape = (self.batch_size, self.dnn_latent_dim)
		eps = K.random_normal(shape = batch_shape, mean = 0., stddev = 1.0)

		gamma_ = K.exp(self.dnn_latent_log_var/2)*eps
		dnn_norm = self.dnn_latent_mean + gamma_
		
		# need to add 0's so we can sum it all to 1
		padding = K.tf.zeros(self.batch_size, 1)[:,None]
		dnn_norm = concatenate([dnn_norm, padding], 
								name='dnn_norm')
		sum_exp_dnn_norm = K.sum(K.exp(dnn_norm), axis = -1)[:,None]
		return K.exp(dnn_norm)/sum_exp_dnn_norm

	def get_model(self, num_gpus = 0, batch_size = None, original_dim = None, 
				  vae_hidden_dims = None, vae_latent_dim = None, 
				  dnn_hidden_dims = None, use_prev_input = False, 
				  dnn_weight = 1.0, vae_weight = 1.0, vae_kl_weight = 1.0, 
				  dnn_kl_weight = 1.0, dnn_log_var_prior = 0.0, 
				  hidden_activation = 'relu', output_activation = 'sigmoid'):

		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		
		if dnn_log_var_prior is not None:
			self.dnn_log_var_prior = dnn_log_var_prior
		
		# update new input values from local args
		if batch_size is not None: self.batch_size = batch_size
		if original_dim is not None: self.original_dim = original_dim
		if vae_hidden_dims is not None: self.vae_hidden_dims = vae_hidden_dims
		if vae_latent_dim is not None: self.vae_latent_dim = vae_latent_dim

		if dnn_hidden_dims is not None: 
			self.dnn_hidden_dims = dnn_hidden_dims
		
		if use_prev_input is not None: self.use_prev_input = use_prev_input

		batch_shape = (self.batch_size, self.original_dim)
		# batch_shape = (self.original_dim,)
		self.input_layer = layers.Input(batch_shape = batch_shape, name='input_layer')

		if use_prev_input or self.use_prev_input:
			self.prev_input_layer = layers.Input(batch_shape = batch_shape, 
										name = 'previous_input_layer')
		self.build_predictor()
		
		self.input_w_pred = concatenate([self.input_layer, 
										self.dnn_latent_layer], 
										axis = -1,
										name = 'data_input_w_dnn_latent_out')

		self.build_latent_encoder()
		
		if use_prev_input or self.use_prev_input:
			self.prev_w_vae_latent = concatenate(
				[self.prev_input_layer, self.vae_latent_layer], 
				 axis = -1, name = 'prev_inp_w_vae_lat_layer')
		else:
			self.prev_w_vae_latent = self.vae_latent_layer
		
		self.dnn_w_latent = concatenate(
						[self.dnn_latent_layer, self.prev_w_vae_latent],
						axis = -1, name = 'dnn_latent_out_w_prev_w_vae_lat')
		
		self.build_latent_decoder()
		
		if use_prev_input or self.use_prev_input:
			input_stack = [self.input_layer, self.prev_input_layer]
			out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
						 self.dnn_latent_mod, self.vae_latent_args]
			enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]
		else:
			input_stack = [self.input_layer]
			out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
						 self.dnn_latent_mod, self.vae_latent_args]
			enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]

		self.model = Model(input_stack, out_stack)
		self.enc_model = Model(input_stack, enc_stack)

		#Make The Model Parallel Using Multiple GPUs
		#if(num_gpus >= 2):
		#	self.model = multi_gpu_model(self.model, gpus=num_gpus)

		self.model.compile(  
				optimizer = self.optimizer,

				loss = {'vae_reconstruction': self.vae_loss,
						'predictor_latent_layer': self.dnn_kl_loss,
						'predictor_latent_mod': self.dnn_rec_loss,
						'vae_latent_args': self.vae_kl_loss},

				loss_weights = {'vae_reconstruction': vae_weight,
								'predictor_latent_layer': dnn_kl_weight,
								'predictor_latent_mod':dnn_weight,
								'vae_latent_args': vae_kl_weight},

				 # metrics=['acc', 'mse'])
				metrics = {'predictor_latent_layer': ['acc', 'mse']})
	
	def vae_sampling(self, args):
		eps = K.random_normal(shape = (self.batch_size, self.vae_latent_dim), 
								mean = 0., stddev = 1.0)
		
		return self.vae_latent_mean + K.exp(self.vae_latent_log_var/2) * eps

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
				from `.yaml` files that has to do with `layers.Lambda` calls
				... so this is a hack for now
		'''
		self.get_model()
		self.model.load_weights(model_file)

	def make_dnn_predictor(self):
		batch_shape = (self.batch_size, self.original_dim)
		# batch_shape = (self.original_dim,)
		input_layer = layers.Input(batch_shape = batch_shape, name = 'input_layer')

		# build label encoder
		enc_hidden_layer = self.model.get_layer('predictor_hidden_layer')
		enc_hidden_layer = enc_hidden_layer(input_layer)
		
		dnn_latent_mean = self.model.get_layer('predictor_latent_mean')
		dnn_latent_mean = dnn_latent_mean(enc_hidden_layer)

		dnn_latent_log_var = self.model.get_layer('predictor_latent_log_var')
		dnn_latent_log_var = dnn_latent_log_var(enc_hidden_layer)

		return Model(input_layer, [dnn_latent_mean, dnn_latent_log_var])

	def make_latent_encoder(self):
		orig_batch_shape = (self.batch_size, self.original_dim)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		# orig_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.predictor_out_dim,)

		input_layer = layers.Input(batch_shape = orig_batch_shape, 
							name = 'input_layer')
		dnn_input_layer = layers.Input(batch_shape = predictor_batch_shape, 
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

	def make_latent_decoder(self, use_prev_input=False):

		input_batch_shape = (self.batch_size, self.original_dim)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		vae_batch_shape = (self.batch_size, self.vae_latent_dim)
		# input_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.dnn_out_dim,)
		# vae_batch_shape = (self.vae_latent_dim,)

		dnn_input_layer = layers.Input(batch_shape = predictor_batch_shape, 
										name = 'predictor_layer')
		vae_latent_layer = layers.Input(batch_shape = vae_batch_shape, 
									name = 'vae_latent_layer')

		if use_prev_input or self.use_prev_input:
			prev_input_layer = layers.Input(batch_shape = input_batch_shape, 
										name = 'prev_input_layer')

		if use_prev_input or self.use_prev_input:
			prev_vae_stack = [prev_input_layer, vae_latent_layer]
			prev_w_vae_latent = concatenate(prev_vae_stack, axis = -1)
		else:
			prev_w_vae_latent = vae_latent_layer

		dnn_w_latent = concatenate([dnn_input_layer,prev_w_vae_latent],
										axis = -1)

		# build physical decoder
		vae_reconstruction = self.model.get_layer('vae_reconstruction')
		if self.vae_hidden_dim > 0:
			vae_dec_hid_layer = self.model.get_layer('vae_dec_hid_layer')
			vae_dec_hid_layer = vae_dec_hid_layer(dnn_w_latent)
			vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)
		else:
			vae_reconstruction = vae_reconstruction(dnn_w_latent)

		if use_prev_input or self.use_prev_input:
			dec_input_stack = [dnn_input_layer, 
								vae_latent_layer, 
								prev_input_layer]
		else:
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

class ConVAEPredictor(object):
	def __init__(self, original_dim, vae_hidden_dims, dnn_hidden_dims, 
					vae_latent_dim, dnn_out_dim = None, 
					dnn_latent_dim = None, batch_size = 128, 
					dnn_log_var_prior = 0.0, optimizer = 'adam-wn', 
					use_prev_input = False, predictor_type = 'classification'):
		
		self.predictor_type = predictor_type
		self.original_dim = original_dim
		
		self.vae_hidden_dims = vae_hidden_dims
		self.vae_latent_dim = vae_latent_dim

		self.dnn_hidden_dims = dnn_hidden_dims
		self.dnn_out_dim = dnn_out_dim
		
		"""FINDME: Why is this dnn_out_dim-1(??)"""
		if dnn_latent_dim is not None:
			self.dnn_latent_dim = self.dnn_out_dim - 1

		self.dnn_log_var_prior = dnn_log_var_prior
		self.optimizer = optimizer
		self.batch_size = batch_size
		self.use_prev_input = use_prev_input
	
	def build_predictor(self):
		if bool(sum(self.dnn_hidden_dims)):
			''' Establish VAE Encoder layer structure '''
			dnn_hidden_layer = build_hidden_layers(Layer = 'Conv1D',
									hidden_dims = self.dnn_hidden_dims, 
									input_layer = self.input_layer, 
									base_layer_name='predictor_hidden_layer', 
									activation=self.hidden_activation)
		else:
			'''if there are no hidden layers, then the input to the 
				dnn latent layers is the input_layer'''
			dnn_hidden_layer = self.input_layer
		
		# Process the predictor layer through a latent layer
		dnn_latent_mean = layers.Dense(self.dnn_latent_dim, 
										name = 'predictor_latent_mean')
		self.dnn_latent_mean = dnn_latent_mean(dnn_hidden_layer)
		dnn_latent_log_var = layers.Dense(self.dnn_latent_dim, 
									name = 'predictor_latent_log_var')
		
		self.dnn_latent_log_var = dnn_latent_log_var(
													dnn_hidden_layer)
		
		dnn_latent_layer = layers.Lambda(self.dnn_sampling, 
									name = 'predictor_latent_layer')
		self.dnn_latent_layer = dnn_latent_layer(
							[self.dnn_latent_mean, self.dnn_latent_log_var])
		
		# Add some wiggle to the dnn predictions 
		#   to avoid division by zero
		dnn_latent_mod = layers.Lambda(lambda tmp: tmp+1e-10, 
								name = 'predictor_latent_mod')
		self.dnn_latent_mod = dnn_latent_mod(self.dnn_latent_layer)
		
	def build_latent_decoder(self):
		shape = K.int_shape(x)
		
		if self.layer_type in 'conv2d':
			upscale = Dense(shape[1] * shape[2] * shape[3], activation='relu')(self.dnn_w_latent)
			x = Reshape((shape[1], shape[2], shape[3]))(x)

		if bool(sum(self.vae_hidden_dims)):
			# reverse order for decoder
			vae_dec_hid_layer = build_hidden_conv_layers(
									filter_sizes = self.vae_hidden_filter_size,
									num_filters = self.num_vae_hidden_layers,
									kernel_sizes = self.vae_hidden_kernel_size,
									input_layer = self.dnn_w_latent,
									strides = self.strides,
									base_layer_name = 'vae_dec_hidden_layer',
									activation  = self.hidden_activation,
									Layer = '{}T'.format(self.layer_type))
		else:
			vae_dec_hid_layer = self.dnn_w_latent
		
		output_layer = {'conv1d':layers.Conv1DTranspose,
						'conv2d':layers.Conv2DTranspose,
						'conv3d':layers.Conv3DTranspose}

		one = 1
		vae_reconstruction = output_layer[self.layer_type](filters = one,
								  kernel_size = self.vae_hidden_kernel_size,
								  activation = self.output_activation,
								  padding = 'same',
								  name = 'vae_reconstruction')
		
		self.vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)

	def build_latent_encoder(self):
		if bool(sum(self.vae_hidden_dims)):
			''' Establish VAE Encoder layer structure '''
			vae_enc_hid_layer = build_hidden_layers(Layer = 'Conv1D',
								hidden_dims = self.vae_hidden_dims, 
								input_layer = self.input_w_pred, 
								base_layer_name='vae_enc_hidden_layer', 
								activation=self.hidden_activation)
		else:
			'''if there are no hidden layers, then the input to the 
				vae latent layers is the input_w_pred layer'''
			vae_enc_hid_layer = self.input_w_pred
		
		vae_latent_mean = layers.Dense(self.vae_latent_dim,name='vae_latent_mean')
		self.vae_latent_mean = vae_latent_mean(vae_enc_hid_layer)
		
		vae_latent_log_var =layers.Dense(self.vae_latent_dim,
								  name = 'vae_latent_log_var')
		self.vae_latent_log_var = vae_latent_log_var(vae_enc_hid_layer)
		
		vae_latent_layer = layers.Lambda(self.vae_sampling, name = 'vae_latent_layer')
		self.vae_latent_layer = vae_latent_layer([self.vae_latent_mean, 
												  self.vae_latent_log_var])

		self.vae_latent_args = concatenate([self.vae_latent_mean, 
											self.vae_latent_log_var], 
											axis = -1, 
											name = 'vae_latent_args')
	
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

	def dnn_sampling(self, args):
		''' sample from a logit-normal with params dnn_latent_mean 
				and dnn_latent_log_var
			(n.b. this is very similar to a logistic-normal distribution)
		'''
		batch_shape = (self.batch_size, self.dnn_latent_dim)
		eps = K.random_normal(shape = batch_shape, mean = 0., stddev = 1.0)

		gamma_ = K.exp(self.dnn_latent_log_var/2)*eps
		dnn_norm = self.dnn_latent_mean + gamma_
		
		# need to add 0's so we can sum it all to 1
		padding = K.tf.zeros(self.batch_size, 1)[:,None]
		dnn_norm = concatenate([dnn_norm, padding], 
								name='dnn_norm')
		sum_exp_dnn_norm = K.sum(K.exp(dnn_norm), axis = -1)[:,None]
		return K.exp(dnn_norm)/sum_exp_dnn_norm

	def get_model(self, num_gpus = 0, batch_size = None, original_dim = None, 
				  vae_hidden_dims = None, vae_latent_dim = None, 
				  dnn_hidden_dims = None, use_prev_input = False, 
				  dnn_weight = 1.0, vae_weight = 1.0, vae_kl_weight = 1.0, 
				  dnn_kl_weight = 1.0, dnn_log_var_prior = 0.0, 
				  hidden_activation = 'relu', output_activation = 'sigmoid'):

		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		
		if dnn_log_var_prior is not None:
			self.dnn_log_var_prior = dnn_log_var_prior
		
		# update new input values from local args
		if batch_size is not None: self.batch_size = batch_size
		if original_dim is not None: self.original_dim = original_dim
		if vae_hidden_dims is not None: self.vae_hidden_dims = vae_hidden_dims
		if vae_latent_dim is not None: self.vae_latent_dim = vae_latent_dim

		if dnn_hidden_dims is not None: 
			self.dnn_hidden_dims = dnn_hidden_dims
		
		if use_prev_input is not None: self.use_prev_input = use_prev_input

		batch_shape = (self.batch_size, self.original_dim)
		# batch_shape = (self.original_dim,)
		self.input_layer = layers.Input(batch_shape = batch_shape, name='input_layer')

		if use_prev_input or self.use_prev_input:
			self.prev_input_layer = layers.Input(batch_shape = batch_shape, 
										name = 'previous_input_layer')
		self.build_predictor()
		
		self.input_w_pred = concatenate([self.input_layer, 
										self.dnn_latent_layer], 
										axis = -1,
										name = 'data_input_w_dnn_latent_out')

		self.build_latent_encoder()
		
		if use_prev_input or self.use_prev_input:
			self.prev_w_vae_latent = concatenate(
				[self.prev_input_layer, self.vae_latent_layer], 
				 axis = -1, name = 'prev_inp_w_vae_lat_layer')
		else:
			self.prev_w_vae_latent = self.vae_latent_layer
		
		self.dnn_w_latent = concatenate(
						[self.dnn_latent_layer, self.prev_w_vae_latent],
						axis = -1, name = 'dnn_latent_out_w_prev_w_vae_lat')
		
		self.build_latent_decoder()
		
		if use_prev_input or self.use_prev_input:
			input_stack = [self.input_layer, self.prev_input_layer]
			out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
						 self.dnn_latent_mod, self.vae_latent_args]
			enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]
		else:
			input_stack = [self.input_layer]
			out_stack = [self.vae_reconstruction, self.dnn_latent_layer, 
						 self.dnn_latent_mod, self.vae_latent_args]
			enc_stack = [self.vae_latent_mean, self.dnn_latent_mean]

		self.model = Model(input_stack, out_stack)
		self.enc_model = Model(input_stack, enc_stack)

		#Make The Model Parallel Using Multiple GPUs
		#if(num_gpus >= 2):
		#	self.model = multi_gpu_model(self.model, gpus=num_gpus)

		self.model.compile(  
				optimizer = self.optimizer,

				loss = {'vae_reconstruction': self.vae_loss,
						'predictor_latent_layer': self.dnn_kl_loss,
						'predictor_latent_mod': self.dnn_rec_loss,
						'vae_latent_args': self.vae_kl_loss},

				loss_weights = {'vae_reconstruction': vae_weight,
								'predictor_latent_layer': dnn_kl_weight,
								'predictor_latent_mod':dnn_weight,
								'vae_latent_args': vae_kl_weight},

				 # metrics=['acc', 'mse'])
				metrics = {'predictor_latent_layer': ['acc', 'mse']})
	
	def vae_sampling(self, args):
		eps = K.random_normal(shape = (self.batch_size, self.vae_latent_dim), 
								mean = 0., stddev = 1.0)
		
		return self.vae_latent_mean + K.exp(self.vae_latent_log_var/2) * eps

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
				from `.yaml` files that has to do with `layers.Lambda` calls
				... so this is a hack for now
		'''
		self.get_model()
		self.model.load_weights(model_file)

	def make_dnn_predictor(self):
		batch_shape = (self.batch_size, self.original_dim)
		# batch_shape = (self.original_dim,)
		input_layer = layers.Input(batch_shape = batch_shape, name = 'input_layer')

		# build label encoder
		enc_hidden_layer = self.model.get_layer('predictor_hidden_layer')
		enc_hidden_layer = enc_hidden_layer(input_layer)
		
		dnn_latent_mean = self.model.get_layer('predictor_latent_mean')
		dnn_latent_mean = dnn_latent_mean(enc_hidden_layer)

		dnn_latent_log_var = self.model.get_layer('predictor_latent_log_var')
		dnn_latent_log_var = dnn_latent_log_var(enc_hidden_layer)

		return Model(input_layer, [dnn_latent_mean, dnn_latent_log_var])

	def make_latent_encoder(self):
		orig_batch_shape = (self.batch_size, self.original_dim)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		# orig_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.predictor_out_dim,)

		input_layer = layers.Input(batch_shape = orig_batch_shape, 
							name = 'input_layer')
		dnn_input_layer = layers.Input(batch_shape = predictor_batch_shape, 
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

	def make_latent_decoder(self, use_prev_input=False):

		input_batch_shape = (self.batch_size, self.original_dim)
		predictor_batch_shape = (self.batch_size, self.dnn_out_dim)
		vae_batch_shape = (self.batch_size, self.vae_latent_dim)
		# input_batch_shape = (self.original_dim,)
		# predictor_batch_shape = (self.dnn_out_dim,)
		# vae_batch_shape = (self.vae_latent_dim,)

		dnn_input_layer = layers.Input(batch_shape = predictor_batch_shape, 
										name = 'predictor_layer')
		vae_latent_layer = layers.Input(batch_shape = vae_batch_shape, 
									name = 'vae_latent_layer')

		if use_prev_input or self.use_prev_input:
			prev_input_layer = layers.Input(batch_shape = input_batch_shape, 
										name = 'prev_input_layer')

		if use_prev_input or self.use_prev_input:
			prev_vae_stack = [prev_input_layer, vae_latent_layer]
			prev_w_vae_latent = concatenate(prev_vae_stack, axis = -1)
		else:
			prev_w_vae_latent = vae_latent_layer

		dnn_w_latent = concatenate([dnn_input_layer,prev_w_vae_latent],
										axis = -1)

		# build physical decoder
		vae_reconstruction = self.model.get_layer('vae_reconstruction')
		if self.vae_hidden_dim > 0:
			vae_dec_hid_layer = self.model.get_layer('vae_dec_hid_layer')
			vae_dec_hid_layer = vae_dec_hid_layer(dnn_w_latent)
			vae_reconstruction = vae_reconstruction(vae_dec_hid_layer)
		else:
			vae_reconstruction = vae_reconstruction(dnn_w_latent)

		if use_prev_input or self.use_prev_input:
			dec_input_stack = [dnn_input_layer, 
								vae_latent_layer, 
								prev_input_layer]
		else:
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
