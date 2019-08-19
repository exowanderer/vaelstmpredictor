import numpy as np
import os
import tensorflow as tf

from contextlib import redirect_stdout

from keras import backend as K
from keras.utils import to_categorical

import joblib
from time import time

from vaelstmpredictor.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmpredictor.utils.model_utils import save_model_in_pieces
from vaelstmpredictor.utils.model_utils import AnnealLossWeight
from vaelstmpredictor.utils.data_utils import MNISTData
from vaelstmpredictor.utils.weightnorm import data_based_init
from vaelstmpredictor.vae_conv1d_predictor.model import ConvVAEPredictor

from keras.backend.tensorflow_backend import set_session

def debug_message(message, end = '\n'): 
	print('[DEBUG] {}'.format(message), end = end)
def info_message(message, end = '\n'): 
	print('[INFO] {}'.format(message), end = end)

class Chromosome(ConvVAEPredictor):
	
	def __init__(self, clargs, data_instance, vae_latent_dim, 
				vae_hidden_dims, dnn_hidden_dims,
				num_conv_layers, size_kernel, size_filter, size_pool = 1,
				#vae_filter_size, dnn_filter_size, 
				#vae_kernel_size = 3, dnn_kernel_size = 3,
				dnn_strides = 1, vae_strides = 1, encoder_top_size = 16, 
				final_kernel_size = 3, data_shape = (784,1), 
				generationID = 0, chromosomeID = 0, 
				vae_kl_weight = 1.0, vae_weight = 1.0, 
				dnn_weight = 1.0, dnn_kl_weight = 1.0,
				dnn_latent_dim = None, batch_size = 128, 
				dnn_log_var_prior = 0.0, save_model = False, 
				n_channels = 1, verbose = False):

		''' Configure dnn '''
		if(isinstance(size_filter, int)):
			self.dnn_filters = [size_filter]*num_conv_layers
			self.encoder_filters = [size_filter]*num_conv_layers
			self.decoder_filters = [size_filter]*num_conv_layers
		else:
			self.dnn_filters = size_filter
			self.encoder_filters = size_filter
			self.decoder_filters = size_filter
		
		if(isinstance(size_kernel, int)):
			self.dnn_kernel_sizes = [size_kernel*2 +1]*num_conv_layers
			self.encoder_kernel_sizes = [size_kernel*2 +1]*num_conv_layers
			self.decoder_kernel_sizes = [size_kernel*2 +1]*num_conv_layers
		else:
			self.dnn_kernel_sizes = size_kernel*2 +1
			self.encoder_kernel_sizes = size_kernel*2 +1
			self.decoder_kernel_sizes = size_kernel*2 +1

		if(isinstance(size_pool, int)):
			self.dnn_pool_sizes = [size_pool*2]*num_conv_layers
			self.encoder_pool_sizes = [size_pool*2]*num_conv_layers
			self.decoder_pool_sizes = [size_pool*2]*num_conv_layers
		else:
			self.dnn_pool_sizes = size_pool*2
			self.encoder_pool_sizes = size_pool*2
			self.decoder_pool_sizes = size_pool*2

		if(isinstance(dnn_strides, int)):
			self.dnn_strides = [dnn_strides]*num_conv_layers
			self.encoder_strides = [vae_strides]*num_conv_layers
			self.decoder_strides = [vae_strides]*num_conv_layers
		else:
			self.dnn_strides = dnn_strides
			self.encoder_strides = vae_strides
			self.decoder_strides = vae_strides

		''' Store in `self` '''
		self.data_shape = data_shape
		
		self.encoder_top_size = encoder_top_size
		
		self.final_kernel_size = final_kernel_size

		"""FINDME: Why is this dnn_out_dim-1(??)"""
		if dnn_latent_dim is not None: 
			self.dnn_latent_dim = dnn_latent_dim
		else:
			self.dnn_latent_dim = clargs.n_labels - 1

		self.dnn_log_var_prior = dnn_log_var_prior
	
		config = tf.ConfigProto()
		# dynamically grow the memory used on the GPU
		config.gpu_options.allow_growth = True

		# to log device placement (on which device the operation ran)
		# (nothing gets printed in Jupyter, only if you run it standalone)
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# set this TensorFlow session as the default session for Keras
		set_session(sess)
		
		self.verbose = verbose
		self.n_channels = n_channels # default for 1D Time-series
		self.save_model = save_model
		self.clargs = clargs
		self.data_instance = data_instance
		self.generationID = generationID
		self.chromosomeID = chromosomeID
		self.time_stamp = clargs.time_stamp
		
		self.vae_latent_dim = vae_latent_dim
		self.vae_kl_weight = vae_kl_weight
		self.vae_weight = vae_weight
		self.dnn_weight = dnn_weight
		self.dnn_kl_weight = dnn_kl_weight
		
		self.vae_hidden_dims = vae_hidden_dims
		self.dnn_hidden_dims = dnn_hidden_dims
		
		'''
		self.params_dict = {}
		for k, layer_size in enumerate(self.vae_hidden_dims):
			self.params_dict['size_vae_hidden{}'.format(k)] = layer_size

		self.params_dict['vae_latent_dim'] = self.vae_latent_dim

		for k, layer_size in enumerate(self.dnn_hidden_dims):
			self.params_dict['size_dnn_hidden{}'.format(k)] = layer_size
		'''
		self.model_dir = clargs.model_dir
		self.run_name = clargs.run_name
		self.original_dim = clargs.original_dim
		self.dnn_weight = clargs.dnn_weight
		
		self.optimizer = clargs.optimizer
		self.batch_size = clargs.batch_size
		self.use_prev_input = False
		self.dnn_out_dim = clargs.n_labels
		
		self.build_model(vae_kl_weight = self.vae_kl_weight,
							vae_weight = self.vae_weight,
							dnn_weight = self.dnn_weight,
							dnn_kl_weight = self.dnn_kl_weight)
		# self.model.compile(optimizer=self.optimizer)
		self.neural_net = self.model
		self.fitness = 0
		self.best_loss = None
		self.isTrained = False
		
		assert(os.path.exists(self.model_dir)), \
				"{} does not exist.".format(self.model_dir) 
		self.topology_savefile = '{}/{}_{}_{}_topology_savefile_{}.save'
		self.topology_savefile = self.topology_savefile.format(
							self.model_dir, self.run_name, self.generationID, 
							self.chromosomeID, self.time_stamp)

		joblib_save_loc ='{}/{}_{}_{}_trained_model_output_{}.joblib.save'
		self.joblib_save_loc = joblib_save_loc.format(self.model_dir, 
										self.run_name, self.generationID, 
										self.chromosomeID, self.time_stamp)

		wghts_save_loc = '{}/{}_{}_{}_trained_model_weights_{}.save'
		self.wghts_save_loc = wghts_save_loc.format(self.model_dir, 
										self.run_name, self.generationID, 
										self.chromosomeID, self.time_stamp)
		
		model_save_loc = '{}/{}_{}_{}_trained_model_full_{}.save'
		self.model_save_loc = model_save_loc.format(self.model_dir, 
										self.run_name, self.generationID, 
										self.chromosomeID, self.time_stamp)

		if verbose: self.neural_net.summary()

	def train(self, verbose = False):
		"""Training control operations to create VAEPredictor instance, 
			organize the input data, and train the network.
		
		Args:
			clargs (object): command line arguments from `argparse`
				Structure Contents: n_labels,
					run_name, patience, kl_anneal, do_log, do_chkpt, num_epochs
					w_kl_anneal, optimizer, batch_size
			
			data_instance (object): 
				Object instance for organizing data structures
				Structure Contents: train_labels, valid_labels, test_labels
					labels_train, data_train, labels_valid, data_valid
		"""
		start_train = time()
		verbose = verbose or self.verbose
		
		DI = self.data_instance
		
		predictor_train = to_categorical(DI.train_labels, self.clargs.n_labels)
		predictor_validation = to_categorical(DI.valid_labels,
											self.clargs.n_labels)

		# Need to expand dimensions to follow Conv1D syntax
		DI.data_train = np.expand_dims(DI.data_train, axis=2)
		DI.data_test = np.expand_dims(DI.data_test, axis=2)
		DI.data_valid = np.expand_dims(DI.data_valid, axis=2)
		DI.labels_train = np.expand_dims(DI.labels_train, axis=2)
		DI.labels_valid = np.expand_dims(DI.labels_valid, axis=2)
		predictor_train = np.expand_dims(predictor_train, axis=2)
		predictor_validation = np.expand_dims(predictor_validation, axis=2)
		
		min_epoch = max(self.clargs.kl_anneal, self.clargs.w_kl_anneal)+1
		callbacks = get_callbacks(self.clargs, patience=self.clargs.patience, 
					min_epoch = min_epoch, do_log = self.clargs.do_log, 
					do_ckpt = self.clargs.do_ckpt)

		if self.clargs.kl_anneal > 0: 
			self.vae_kl_weight = K.variable(value=0.1)
		if self.clargs.w_kl_anneal > 0: 
			self.dnn_kl_weight = K.variable(value=0.0)
		
		if self.save_model: save_model_in_pieces(self.model, self.clargs)
		
		vae_train = DI.data_train
		vae_features_val = DI.data_valid

		data_based_init(self.model, DI.data_train[:self.clargs.batch_size])

		vae_labels_val = [DI.labels_valid, predictor_validation, 
							predictor_validation,DI.labels_valid]

		validation_data = (vae_features_val, vae_labels_val)
		train_labels = [DI.labels_train, predictor_train, 
						predictor_train, DI.labels_train]
		
		print('\n\nFITTING MODEL\n\n')

		self.history = self.model.fit(vae_train, train_labels,
									shuffle = True,
									epochs = self.clargs.num_epochs,
									batch_size = self.clargs.batch_size,
									callbacks = callbacks,
									validation_data = validation_data)

		max_kl_anneal = max(self.clargs.kl_anneal, self.clargs.w_kl_anneal)
		self.best_ind = np.argmin([x if i >= max_kl_anneal + 1 else np.inf \
					for i,x in enumerate(self.history.history['val_loss'])])
		
		self.best_loss = {k: self.history.history[k][self.best_ind] \
										for k in self.history.history}
		
		# self.best_val_loss = sum([val for key,val in self.best_loss.items() \
		#							 if 'val_' in key and 'loss' in key])
		
		self.fitness = 1.0 / self.best_loss['val_loss']
		self.isTrained = True
		
		if verbose: 
			print('\n\n')
			print("Generation: {}".format(self.generationID))
			print("Chromosome: {}".format(self.chromosomeID))
			print("Operation Time: {}".format(time() - start_train))
			print('\nBest Loss:')
			for key,val in self.best_loss.items():
				print('{}: {}'.format(key,val))

			print('\nFitness: {}'.format(self.fitness))
			print('\n\n')
		
		if self.save_model: self.save()

	def save(self):
		# Save network topology
		with open(self.topology_savefile, 'w') as f:
			with redirect_stdout(f):
				self.neural_net.summary()

		yaml_filename = self.topology_savefile.replace('.save', '.yaml')
		with open(yaml_filename, 'w') as yaml_fileout:
			yaml_fileout.write(self.neural_net.to_yaml())
		
		# save model args
		json_filename = self.topology_savefile.replace('.save', '.json')
		with open(json_filename, 'w') as json_fileout:
			json_fileout.write(self.neural_net.to_json())

		# Save network weights
		self.neural_net.save_weights(self.wghts_save_loc, overwrite=True)

		# Save network entirely
		self.neural_net.save(self.model_save_loc, overwrite=True)

		# Save class object in its entirety
		try:
			joblib.dump({'best_loss':self.best_loss,
							'history':self.history}, 
							self.joblib_save_loc)
		except Exception as e:
			print(str(e))
