# from https://github.com/exowanderer/vaelstmpredictor/blob/
#	GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 
#	--verbose --num_generations 500 --population_size 10 --num_epochs 200
import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import requests

from contextlib import redirect_stdout
from glob import glob
from keras import backend as K
from keras.utils import to_categorical
from numpy import array, arange, vstack, reshape, loadtxt, zeros, random
from paramiko import SSHClient, SFTPClient, Transport, AutoAddPolicy, ECDSAKey
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmpredictor.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmpredictor.utils.model_utils import save_model_in_pieces
from vaelstmpredictor.utils.model_utils import AnnealLossWeight
from vaelstmpredictor.utils.data_utils import MNISTData
from vaelstmpredictor.utils.weightnorm import data_based_init
from vaelstmpredictor.vae_predictor.model import VAEPredictor
from vaelstmpredictor.vae_predictor.train import train_vae_predictor

from .Chromosome import Chromosome

def configure_multi_hidden_layers(num_hidden, input_size, 
								  min_hidden1, max_hidden,
								  start_small = True, init_large = True):
	# To force outer boundary inclusion with numpy
	max_hidden = max_hidden + 1 
	input_size = input_size + 1

	zero = 0
	if input_size is not None and init_large:
		hidden_dims = [random.randint(input_size//2, input_size)]
	else:
		hidden_dims = [random.randint(min_hidden1, max_hidden)]
	
	# set to zero or random
	if start_small:
		hidden_dims = hidden_dims + [zero]*(num_hidden-1)
	else:
		upper_layers = np.random.randint(zero, max_hidden, size = num_hidden-1)

		hidden_dims = np.append(hidden_dims, upper_layers)

	return hidden_dims

def query_sql_database(clargs, chromosome):
	getFitness = 'http://LAUDeepGenerativeGenetics.pythonanywhere.com/GetFitness'

	table_dir = clargs.table_dir
	table_name = '{}/{}_{}_{}_sql_fitness_table_{}.json'
	table_name = table_name.format(clargs.table_dir, 
						clargs.run_name, chromosome.generationID, 
						chromosome.chromosomeID, clargs.time_stamp)

	json_ID = {'generationID':int(chromosome.generationID),
			   'chromosomeID':int(chromosome.chromosomeID)}
	
	sql_json = requests.get(getFitness, params=json_ID).json()
	
	if not isinstance(sql_json, dict):
		print('SQL Request Failed: sql_json = {} with {}'.format(
												sql_json, json_ID))
		return -1

	with open(table_name, 'a') as f_out:
		json.dump(sql_json, f_out)
	
	return sql_json['fitness']
	# 	chromosome.fitness = sql_json['fitness']
	# else:
	# 	print('SQL Request Failed: sql_json = {} with {}'.format(
	# 											sql_json, json_ID))
	# 	chromosome.fitness = sql_json or 0
	# 
	# return chromosome.fitness

def query_local_csv(clargs, chromosome):
	
	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(clargs.table_dir, 
									clargs.run_name, 
									clargs.time_stamp)

	generationID = int(chromosome.generationID)
	chromosomeID = int(chromosome.chromosomeID)
	if os.path.exists(table_name):
		with open(table_name, 'r') as f_in:
			check = 'fitness:'
			for line in f_in.readlines():
				ck1 = 'generationID:{}'.format(generationID) in line
				ck2 = 'chromosomeID:{}'.format(chromosomeID) in line
				if ck2 and ck2:
					fitness = line.split(check)[1].split(',')[0]
					return float(fitness)

	return -1

def generate_random_chromosomes(population_size,# clargs, data_instance, 
						min_vae_hidden_layers = 1, min_dnn_hidden_layers = 1, 
						max_vae_hidden_layers = 5, max_dnn_hidden_layers = 5, 
						min_vae_hidden = 2, max_vae_hidden = 1024, 
						min_dnn_hidden = 2, max_dnn_hidden = 1024, 
						min_vae_latent = 2, max_vae_latent = 1024, 
						# vae_weight = 1.0, vae_kl_weight = 1.0, 
						# dnn_weight = 1.0, dnn_kl_weight = 1.0, 
						# input_size = None, TrainFunction = None,
						verbose=False):
	# start_small = False, init_large = False, # In kwargs
	# start_small = start_small or clargs.start_small 
	# init_large = init_large or clargs.init_large
	# max_vae_hidden_layers = max_vae_hidden_layers \
	#							 or clargs.max_vae_hidden_layers 
	# max_dnn_hidden_layers = max_dnn_hidden_layers \
	#							 or clargs.max_dnn_hidden_layers 
	# min_vae_hidden = min_vae_hidden or clargs.min_vae_hidden 
	# min_vae_latent = min_vae_latent or clargs.min_vae_latent 
	# min_dnn_hidden = min_dnn_hidden or clargs.min_dnn_hidden 
	# max_vae_hidden = max_vae_hidden or clargs.max_vae_hidden 
	# max_vae_latent = max_vae_latent or clargs.max_vae_latent 
	# max_dnn_hidden = max_dnn_hidden or clargs.max_dnn_hidden 
	# vae_kl_weight = vae_kl_weight or clargs.vae_kl_weight 
	# input_size  = input_size or clargs.original_dim 
	# dnn_weight = dnn_weight or clargs.dnn_weight 
	# dnn_kl_weight = dnn_kl_weight or clargs.dnn_kl_weight 
	# verbose = verbose or clargs.verbose

	# generationID = 0
	# generation_0 = []

	vae_nLayers_choices = range(min_vae_hidden_layers, max_vae_hidden_layers)
	dnn_nLayers_choices = range(min_dnn_hidden_layers, max_dnn_hidden_layers)
	vae_latent_choices = range(min_vae_latent, max_vae_latent)
	vae_nUnits_choices = range(min_vae_hidden, max_vae_hidden)
	dnn_nUnits_choices = range(min_dnn_hidden, max_dnn_hidden)
	
	generation = pd.DataFrame()
	generation['generationID'] = np.zeros(population_size, dtype = int)
	generation['chromosomeID'] = np.arange(population_size, dtype = int)
	generation['isTrained'] = np.zeros(population_size, dtype = bool)
	generation['num_vae_layers'] = np.random.choice(vae_nLayers_choices,
														size = population_size)
	generation['num_dnn_layers'] = np.random.choice(dnn_nLayers_choices,
														size = population_size)
	generation['size_vae_latent'] = np.random.choice(vae_latent_choices, 
														size = population_size)
	generation['size_vae_hidden'] = np.random.choice(vae_nUnits_choices, 
														size = population_size)
	generation['size_dnn_hidden'] = np.random.choice(dnn_nUnits_choices, 
														size = population_size)
	generation['fitness'] = np.zeros(population_size, dtype = int) - 1

	return generation
	# if(TrainFunction is None):

	#	 for chromosome in generation_0:
	#		 chromosome.train()
	# else:
	#	 TrainFunction(generation_0, clargs)
		
	# return generation_0

def train_generation(generation, clargs, private_key='id_ecdsa'):
	getChrom = 'https://LAUDeepGenerativeGenetics.pythonanywhere.com/GetChrom'
	
	key_filename = os.environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	machines = [# {"host": "192.168.0.1", "username": "acc", 
				#   "key_filename": key_filename},
				{"host": "172.16.50.181", "username": "acc", 
					"key_filename": key_filename},
				# {"host": "172.16.50.176", "username": "acc", 
					# "key_filename": key_filename},
				{"host": "172.16.50.177", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.163", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.182", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.218", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.159", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.235", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.157", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.237", "username": "acc", 
					"key_filename": key_filename}
				]
	
	queue = mp.Queue()
	
	#Create Processes
	for machine in machines: queue.put(machine)
	
	alldone = False
	while not alldone:
		alldone = True
		for k, chromosome in generation.iterrows():
			if not chromosome.isTrained:
				print("Creating Process for Chromosome {}".format(
							chromosome.chromosomeID), end=" on machine ")
				# Find a Chromosome that is not trained yet
				alldone = False
				
				# Wait for queue to have a value, 
				#	which is the ID of the machine that is done.
				machine = queue.get()
				print('{}'.format(machine['host']))
				process = mp.Process(target=train_chromosome, 
									args=(chromosome, machine, queue, clargs))
				process.start()
				
				chromosome.isTrained = 1
				generation.iloc[k] = chromosome # finally, we figured this out!

	for k, chromosome in generation.iterrows():
		assert(chromosome.isTrained), 'while loop should not have closed!'
		chromosome.fitness = query_sql_database(clargs, chromosome)
		
		if chromosome.fitness is -1:
			chromosome.fitness = query_local_csv(clargs, chromosome)

		if chromosome.fitness is not -1:
			print('\n\n[INFO]')
			print('GenerationID:{}'.format(chromosome.generationID))
			print('ChromosomeID:{}'.format(chromosome.chromosomeID))
			print('Fitness:{}'.format(chromosome.fitness))
			print('Num VAE Layers:{}'.format(chromosome.num_vae_layers))
			print('Num DNN Layers:{}'.format(chromosome.num_dnn_layers))
			print('Size VAE Latent:{}'.format(chromosome.size_vae_latent))
			print('Size VAE Hidden:{}'.format(chromosome.size_vae_hidden))
			print('Size DNN Hidden:{}'.format(chromosome.size_dnn_hidden))
			print('\n\n')

			generation.iloc[k] = chromosome
	
	# After all is done: return what you received
	return generation

def generate_ssh_command(clargs, chromosome):
	command = []
	command.append('cd vaelstmpredictor; ')
	command.append('../anaconda3/envs/tf_gpu/bin/python run_chromosome.py ')
	command.append('--run_name {}'.format(clargs.run_name))
	command.append('--predictor_type {}'.format(clargs.predictor_type))
	command.append('--batch_size {}'.format(clargs.batch_size))
	command.append('--optimizer {}'.format(clargs.optimizer))
	command.append('--num_epochs {}'.format(clargs.num_epochs))
	command.append('--dnn_weight {}'.format(clargs.dnn_weight))
	command.append('--vae_weight {}'.format(clargs.vae_weight))
	command.append('--vae_kl_weight {}'.format(clargs.vae_kl_weight))
	command.append('--dnn_kl_weight {}'.format(clargs.dnn_kl_weight))
	command.append('--prediction_log_var_prior {}'.format(
											clargs.prediction_log_var_prior))
	# command.append('--do_log {}'.format(int(clargs.do_log)))
	# command.append('--do_chckpt {}'.format(int(clargs.do_chckpt)))
	command.append('--patience {}'.format(clargs.patience))
	command.append('--kl_anneal {}'.format(clargs.kl_anneal))
	command.append('--w_kl_anneal {}'.format(clargs.w_kl_anneal))
	command.append('--dnn_log_var_prior {}'.format(clargs.dnn_log_var_prior))
	command.append('--log_dir {}'.format(clargs.log_dir))
	command.append('--model_dir {}'.format(clargs.model_dir))
	command.append('--table_dir {}'.format(clargs.table_dir))
	command.append('--train_file {}'.format(clargs.train_file))
	command.append('--time_stamp {}'.format(int(clargs.time_stamp)))
	# command.append('--verbose {}'.format(int(clargs.verbose)))
	command.append('--hostname {}'.format(clargs.hostname))
	command.append('--port {}'.format(clargs.port))
	command.append('--num_vae_layers {}'.format(chromosome.num_vae_layers))
	command.append('--num_dnn_layers {}'.format(chromosome.num_dnn_layers))
	command.append('--size_vae_latent {}'.format(chromosome.size_vae_latent))
	command.append('--size_vae_hidden {}'.format(chromosome.size_vae_hidden))
	command.append('--size_dnn_hidden {}'.format(chromosome.size_dnn_hidden))
	command.append('--generationID {} '.format(chromosome.generationID))
	command.append('--chromosomeID {} '.format(chromosome.chromosomeID))
	
	return " ".join(command)

def git_clone(hostname, username = "acc", gitdir = 'vaelstmpredictor',
				gituser = 'exowanderer', 
				branchname = 'MultiComputerGeneticAlgorithm',
				port = 22, verbose = True, private_key='id_ecdsa'):
	
	key_filename = environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(hostname, key_filename = key_filename)

	command = []
	command.append('git clone https://github.com/{}/{}'.format(gituser,gitdir))
	command.append('cd {}'.format(gitdir))
	command.append('git pull')
	command.append('git checkout {}'.format(branchname))
	command.append('git pull')
	command = '; '.join(command)

	print('[INFO] Executing {} on {}'.format(command, hostname))

	stdin, stdout, stderr = ssh.exec_command(command)
	
	try:
		stdout.channel.recv_exit_status()
		for line in stdout.readlines(): print(line)
	except Exception as e:
		print('error on stdout.readlines(): {}'.format(str(e)))

	try:
		stderr.channel.recv_exit_status()
		for line in stderr.readlines(): print(line)
	except Exception as e:
		print('error on stderr.readlines(): {}'.format(str(e)))
	
	print("Command Executed Successfully")
	ssh.close()

def upload_zip_file(zip_filename, machine, verbose = False):
	if verbose: 
			print('[INFO] File {} does not exists on {}'.format(
									zip_filename, machine['host']))
		
	#Upload Files to Machine
	print("Uploading file to machine")
	
	if verbose: 
		print('[INFO] Transfering {} to {}'.format(
							zip_filename, machine['host']))

	transport = Transport((machine["host"], port))
	pk = ECDSAKey.from_private_key(open(machine['key_filename']))
	transport.connect(username = machine["username"], pkey=pk)
	
	sftp = SFTPClient.from_transport(transport)
	sftp.put(zip_filename, zip_filename)
	
	stdin, stdout, stderr = ssh.exec_command('unzip {}'.format(zip_filename))
	
	error = "".join(stderr.readlines())
	if error != "":
		print("Errors has occured while unzipping file in machine: "\
				"{} \nError: {}".format(machine, error))
	
	stdin, stdout, stderr = ssh.exec_command('cd vaelstmpredictor; '
					'../anaconda3/envs/tf_gpu/bin/python setup.py install')
	error = "".join(stderr.readlines())
	if error != "":
		print("Errors setting up vaelstmpredictor: "
				"{}\nError: {}".format(machine, error))

	sftp.close()
	transport.close()
	print("File uploaded")

def train_chromosome(chromosome, machine, queue, clargs, 
					port = 22, logdir = 'train_logs',
					git_dir = 'vaelstmpredictor',
					verbose = True):
	
	if not os.path.exists(logdir): os.mkdir(logdir)

	if verbose: 
		print('[INFO] Checking if file {} exists on {}'.format(
										git_dir, machine['host']))
	
	# chromosomeID = chromosome.chromosomeID
	# sys.stdout = open('{}/output{}.txt'.format(logdir, chromosomeID),'w')
	# sys.stderr = open('{}/error{}.txt'.format(logdir, chromosomeID), 'w')
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(machine["host"], key_filename=machine['key_filename'])

	stdin, stdout, stderr = ssh.exec_command('ls | grep {}'.format(git_dir))
	
	if(len(stdout.readlines()) == 0):
		git_clone()
	elif verbose: 
			print('[INFO] File {} exists on {}'.format(
								git_dir, machine['host']))

	command = generate_ssh_command(clargs, chromosome)
	
	print("\n\nExecuting command:\n\t{}".format(command))
	
	stdin, stdout, stderr = ssh.exec_command(command)
	
	try:
		stdout.channel.recv_exit_status()
		for line in stdout.readlines(): print(line)
	except Exception as e:
		print('error on stdout.readlines(): {}'.format(str(e)))

	try:
		stderr.channel.recv_exit_status()
		for line in stderr.readlines(): print(line)
	except Exception as e:
		print('error on stderr.readlines(): {}'.format(str(e)))
	
	queue.put(machine)

	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(clargs.table_dir, 
									clargs.run_name, 
									clargs.time_stamp)
	"""
	if os.path.exists(table_name):
		with open(table_name, 'r') as f_in:
			check = 'fitness:'
			for line in f_in.readlines():
				ck1 = 'generationID:{}'.format(chromosome.generationID) in line
				ck2 = 'chromosomeID:{}'.format(chromosome.chromosomeID) in line
				if ck2 and ck2:
					fitness = line.split(check)[1].split(',')[0]
					chromosome.fitness = float(fitness)
					break

	chromosome.isTrained = True
	"""
	ssh.close()
	
	print("Command Executed Successfully")

def select_parents(generation):
	total_fitness = sum(chrom.fitness for k, chrom in generation.iterrows())
	#Generate two random numbers between 0 and total_fitness 
	#   not including total_fitness
	rand_parent1 = random.random()*total_fitness
	rand_parent2 = random.random()*total_fitness
	
	parent1 = None
	parent2 = None
	
	fitness_count = 0
	for k, chromosome in generation.iterrows():
		fitness_count += chromosome.fitness
		if(parent1 is None and fitness_count >= rand_parent1):
			parent1 = chromosome
		if(parent2 is None and fitness_count >= rand_parent2):
			parent2 = chromosome
		if(parent1 is not None and parent2 is not None):
			break

	return parent1, parent2

def reconfigure_vae_params(params, static_params_):
	clargs = static_params_['clargs']

	max_vae_hidden_layers = clargs.max_vae_hidden_layers
	max_dnn_hidden_layers = clargs.max_dnn_hidden_layers

	for key,val in static_params_.items():
		params[key] = val

	# Reconfigure child1's collection of layer sizes into arrays
	vae_hidden_dims = np.zeros(max_vae_hidden_layers, dtype=int)
	dnn_hidden_dims = np.zeros(max_dnn_hidden_layers, dtype=int)

	params_copy = {key:val for key,val in params.items()}

	for key,val in params_copy.items():
		if 'size_vae_hidden' in key:
			idx = int(key[-1])
			vae_hidden_dims[idx] = val
			del params[key]

		if 'size_dnn_hidden' in key:
			idx = int(key[-1])
			dnn_hidden_dims[idx] = val
			del params[key]
	
	# params['vae_latent_dim'] = vae_latent_dim
	params['vae_hidden_dims'] = vae_hidden_dims
	params['dnn_hidden_dims'] = dnn_hidden_dims
	
	return params

def cross_over(parent1, parent2, prob, param_choices, verbose=False):
	if verbose: 
		print('[INFO] Crossing over with probability: {}'.format(prob))

	if random.random() >= prob:
		crossover_happened = True
		child = parent1.copy() # this only sets up the pd.Series framework
		for param in param_choices:
			child[param] = random.choice([parent1[param], parent2[param]])
	else: 
		crossover_happened = False
		child = parent1 if parent1.fitness > parent2.fitness else parent2

	return child, crossover_happened
"""
def cross_over_orig(parent1, parent2, prob, verbose=False):
	if verbose:
		print('Crossing over with Parent {} and Parent {}'.format(
						parent1.chromosomeID, parent2.chromosomeID))

	static_params_ = {  'clargs':parent1.clargs, 
						'data_instance':parent1.data_instance, 
						'vae_weight':parent1.vae_weight, 
						'vae_kl_weight':parent1.vae_kl_weight, 
						'dnn_weight':parent1.dnn_weight,
						'dnn_kl_weight':parent1.dnn_kl_weight,
						'verbose':parent1.verbose
					  }

	crossover_happened = True # Flag if child == parent: no need to train

	if(random.random() <= prob):
		params1 = {}
		params2 = {}
		for param in parent1.params_dict.keys():
			if(random.random() <= 0.5):
				params1[param] = parent1.params_dict[param]
				params2[param] = parent2.params_dict[param]
			else:
				params1[param] = parent2.params_dict[param]
				params2[param] = parent1.params_dict[param]
		
		# Reconfigure each child's collection of layer sizes into arrays
		params1 = reconfigure_vae_params(params1, static_params_)
		params2 = reconfigure_vae_params(params2, static_params_)
		
		# child1 = Chromosome(**params1)
		# child2 = Chromosome(**params2)

		return child1, child2, crossover_happened
	
	return parent1, parent2, not crossover_happened
"""
def mutate(child, prob, param_choices, forced_evolve = False, 
			min_layer_size = 2, verbose = False):
	
	# explicit declaration
	zero = 0 

	if verbose:
		print('Mutating Child {} in Generation {}'.format(child.chromosomeID, 
														 child.generationID))
	
	mutation_happened = False
	for param, (range_change, min_val) in param_choices.items():
		if(random.random() <= prob):
			mutation_happened = True

			# Compute delta_param step
			change_p = np.random.uniform(-range_change, range_change)

			# Add delta_param to param
			current_p = child[param] + change_p
			
			# If param less than `min_val`, then set param to `min_val`
			child[param] = np.max([current_p, min_val])
			
			# All params must be integer sized: round and convert
			child[param] = np.int(np.round(child[param]))

	return child, mutation_happened

def save_generation_to_tree(generation, verbose = False):
	generation_dict = {}
	if verbose: print('[INFO] Current Generation: ' )

	for ID, member in generation.iterrow():
		if ID not in generation_dict.keys(): generation_dict[ID] = {}
		
		if verbose: 
			print('memberID: {}'.format(ID))
			print('Fitness: {}'.format(member.fitness))
			for key,val in member.items():
				print('\t{}: {}'.format(key, val))

		generation_dict[ID]['params'] = member
		generation_dict[ID]['fitness'] = member.fitness
	
	return generation_dict
