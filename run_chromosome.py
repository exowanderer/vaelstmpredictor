# from https://github.com/philippesaade11/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --iterations 500 --population_size 10 --num_epochs 200
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import socket

from GeneticAlgorithm import *

from time import time, sleep
from vaelstmpredictor.utils.data_utils import MNISTData

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_name', type=str, default='ga_test',
				help='tag for current run')
	parser.add_argument('--predictor_type', type=str, default="classification",
				help='select `classification` or `regression`')
	parser.add_argument('--batch_size', type=int, default=128,
				help='batch size')
	parser.add_argument('--optimizer', type=str, default='adam',
				help='optimizer name') 
	parser.add_argument('--num_epochs', type=int, default=200,
				help='number of epochs')
	parser.add_argument('--dnn_weight', type=float, default=1.0,
				help='relative weight on prediction loss')
	parser.add_argument('--vae_weight', type=float, default=1.0,
				help='relative weight on prediction loss')
	parser.add_argument('--vae_kl_weight', type=float, default=1.0,
				help='relative weight on prediction loss')
	parser.add_argument('--dnn_kl_weight', type=float, default=1.0,
				help='relative weight on prediction loss')
	parser.add_argument('--prediction_log_var_prior', type=float, default=0.0,
				help='w log var prior')
	# parser.add_argument("--do_log", action="store_true", 
	# 			help="save log files")
	# parser.add_argument("--do_chckpt", action="store_true",
	# 			help="save model checkpoints")
	parser.add_argument('--patience', type=int, default=10,
				help='# of epochs, for early stopping')
	parser.add_argument("--kl_anneal", type=int, default=0, 
				help="number of epochs before kl loss term is 1.0")
	parser.add_argument("--w_kl_anneal", type=int, default=0, 
				help="number of epochs before w's kl loss term is 1.0")
	parser.add_argument('--dnn_log_var_prior', type=float, default=0.0,
				help='Prior on the log variance for the DNN predictor')
	parser.add_argument('--log_dir', type=str, default='data/logs',
				help='basedir for saving log files')
	parser.add_argument('--model_dir', type=str, default='data/models',
				help='basedir for saving model weights')
	parser.add_argument('--table_dir', type=str, default='data/tables',
				help='basedir for storing the table of params and fitnesses.')
	parser.add_argument('--train_file', type=str, default='MNIST',
				help='file of training data (.pickle)')
	# parser.add_argument('--verbose', action='store_true',
	# 			help='print more [INFO] and [DEBUG] statements')
	parser.add_argument('--hostname', type=str, default='127.0.0.1',
				help='The hostname of the computer to send results back to.')
	parser.add_argument('--port', type=int, default=22,
				help='The port on the work computer to send ssh over.')
	parser.add_argument('--generationID', type=int, default=-1,
				help='Chromosome Generation ID')
	parser.add_argument('--chromosomeID', type=int, default=-1,
				help='Chromosome Chromosome ID')
	parser.add_argument('--num_vae_layers', type=int, default=0,
				help='Depth of the VAE')
	parser.add_argument('--num_dnn_layers', type=int, default=0,
				help='Depth of the DNN')
	parser.add_argument('--size_vae_latent', type=int, default=0,
				help='Size of the VAE Latent Layer')
	parser.add_argument('--size_vae_hidden', type=int, default=0,
				help='Size of the VAE Hidden Layer')
	parser.add_argument('--size_dnn_hidden', type=int, default=0,
				help='Size of the DNN Hidden Layer')

	clargs = parser.parse_args()
	
	clargs.do_log = True
	clargs.do_chckpt = True
	clags.verbose = True

	for key,val in clargs.__dict__.items(): 
		if 'dir' in key: 
			if not os.path.exists(val): 
				os.mkdir(val)

	vae_hidden_dims = [clargs.size_vae_hidden]*clargs.num_vae_layers
	dnn_hidden_dims = [clargs.size_dnn_hidden]*clargs.num_dnn_layers
	
	port = clargs.port

	chrom_params = {}
	chrom_params['verbose'] = clargs.verbose
	chrom_params['vae_hidden_dims'] = vae_hidden_dims
	chrom_params['dnn_hidden_dims'] = dnn_hidden_dims
	chrom_params['vae_latent_dim'] = clargs.size_vae_latent
	chrom_params['batch_size'] = clargs.batch_size
	chrom_params['dnn_log_var_prior'] = clargs.dnn_log_var_prior
	chrom_params['optimizer'] = clargs.optimizer
	chrom_params['predictor_type'] = clargs.predictor_type
	chrom_params['clargs'] = clargs
	chrom_params['generationID'] = clargs.generationID
	chrom_params['chromosomeID'] = clargs.chromosomeID
	chrom_params['vae_weight'] = clargs.vae_weight
	chrom_params['vae_kl_weight'] = clargs.vae_kl_weight
	chrom_params['dnn_weight'] = clargs.dnn_weight
	chrom_params['dnn_kl_weight'] = clargs.dnn_kl_weight

	data_instance = MNISTData(batch_size = clargs.batch_size)
	
	n_train, n_features = data_instance.data_train.shape
	n_test, n_features = data_instance.data_valid.shape

	clargs.original_dim = n_features
	clargs.n_labels = len(np.unique(data_instance.train_labels))
	
	chrom_params['data_instance'] = data_instance
	# chrom_params['original_dim'] = clargs.original_dim
	# chrom_params['dnn_out_dim'] = clargs.n_labels
	# chrom_params['dnn_latent_dim'] = clargs.n_labels - 1
	
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	hostname = s.getsockname()[0]
	s.close()

	print('\n\nParams for this VAE_NN:')
	for key,val in clargs.__dict__.items():
		print('{:20}{}'.format(key,val))

	chrom = Chromosome(**chrom_params)
	chrom.verbose = True
	chrom.train(verbose=True)

	transport = Transport((clargs.hostname, port))
	pk = ECDSAKey.from_private_key(open(machine['key_filename']))
	transport.connect(username = machine["username"], pkey=pk)

	sftp = SFTPClient.from_transport(transport)
	sftp.put(chrom.wghts_save_loc, 
			 'vaelstmpredictor/{}'.format(chrom.wghts_save_loc))

	sftp.close()
	transport.close()

	table_dir = clargs.table_dir
	table_name = '{}/{}_{}_{}_fitness_table_{}.csv'
	table_name = table_name.format(clargs.table_dir, 
		clargs.run_name, clargs.generationID, 
		clargs.chromosomeID, clargs.time_stamp)

	entry = []
	entry.append('generationID:{}'.format(clargs.generationID))
	entry.append('chromosomeID:{}'.format(clargs.chromosomeID))
	entry.append('fitness:{}'.format(chrom.fitness))
	for key,val in clargs.__dict__.items():
		if key not in ['generationID', 'chromosomeID']:
			entry.append('{}:{}'.format(key,val))

	entry = ','.join(entry)

	private_key = os.environ['HOME'] + '/.ssh/{}'.format('id_ecdsa')
	command = 'echo "{}" >> vaelstmpredictor/{}'.format(entry, table_name)
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(hostname, key_filename=private_key)
	stdin, stdout, stderr = ssh.exec_command(command)

	clargs.fitness = chrom.fitness
	
	print('\n[INFO]')
	print('Result: ', end=" ")
	print('GenerationID: {}'.format(clargs.generationID), end=" ")
	print('ChromosomeID: {}'.format(clargs.chromosomeID), end=" ")
	print('Fitness: {}\n'.format(clargs.fitness))
	
	putURL = 'https://philippesaade11.pythonanywhere.com/AddChrom'
	
	print('[INFO] Storing to SQL db at {}'.format(putURL))
	
	try:
		req = requests.get(url = putURL, params = clargs.__dict__)
		if req.json() == 1:
			print('[INFO] Remote SQL Entry Added Successfully')
		else:
			print('[WARNING] !! The World Has Ended !!')
	except Exception as e:
		print('[WARNING] Remote SQL Entry Failed:\n{}'.format(str(e)))
