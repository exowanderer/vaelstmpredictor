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
import subprocess

from contextlib import redirect_stdout
from glob import glob
from keras import backend as K
from keras.utils import to_categorical
from numpy import array, arange, vstack, reshape, loadtxt, zeros, random

import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	from paramiko import SSHClient, SFTPClient, Transport
	from paramiko import AutoAddPolicy, ECDSAKey
	from paramiko.ssh_exception import NoValidConnectionsError

warnings.filterwarnings(action='ignore',module='.*paramiko.*')
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings(action='ignore',module='.*sklearn.*')
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmpredictor.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmpredictor.utils.model_utils import save_model_in_pieces
from vaelstmpredictor.utils.model_utils import AnnealLossWeight
from vaelstmpredictor.utils.data_utils import MNISTData
from vaelstmpredictor.utils.weightnorm import data_based_init
from vaelstmpredictor.vae_predictor.dense_model import VAEPredictor
from vaelstmpredictor.vae_predictor.train import train_vae_predictor

from .Chromosome import Chromosome


def debug_message(message, end = '\n'):
	print('[DEBUG] {}'.format(message), end = end)

def warning_message(message, end = '\n'):
	print('[WARNING] {}'.format(message), end = end)

def info_message(message, end = '\n'):
	print('[INFO] {}'.format(message), end = end)

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

def query_sql_database(generationID, chromosomeID, clargs=None, verbose=True):
	getFitness = 'http://LAUDeepGenerativeGenetics.pythonanywhere.com/'
	getFitness = getFitness + 'GetFitness'
	
	if clargs is not None:
		table_dir = clargs.table_dir
		table_name = '{}/{}_{}_{}_sql_fitness_table_{}.json'
		table_name = table_name.format(clargs.table_dir, 
						clargs.run_name, generationID, 
						chromosomeID, clargs.time_stamp)
		
	json_ID = {'generationID':generationID, 'chromosomeID':chromosomeID}
	
	sql_json = requests.get(getFitness, params=json_ID)
	
	print(getFitness,end="?")
	for key,val in json_ID.items():
		print('{}={}&'.format(key,val), end="")
	print('')

	try:
		sql_json = sql_json.json()
	except Exception as error:
		warning_message('query_sql_database+Except:\n{}'.format(error))
	
	if sql_json == 0:#not isinstance(sql_json, requests.models.Response):
		if verbose: 
			print('SQL Request Failed: sql_json = {} with {}'.format(sql_json, 
																	json_ID))
		return sql_json

	# Only triggered if `sql_json` is a `dict`

	# sql_json = sql_json.json()
	
	if clargs is not None:
		with open(table_name, 'a') as f_out: 
			json.dump(sql_json, f_out)

	return sql_json

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

def create_blank_dataframe(generationID, population_size):

	generation = pd.DataFrame()
	
	zeros = np.zeros(population_size, dtype = int)
	ones = np.ones(population_size, dtype = int)
	arange = np.arange(population_size, dtype = int)

	generation['generationID'] = ones.copy()
	generation['chromosomeID'] = arange.copy()
	generation['isTrained'] = zeros.copy()
	generation['num_vae_layers'] = zeros.copy()
	generation['num_dnn_layers'] = zeros.copy()
	generation['size_vae_latent'] = zeros.copy()
	generation['size_vae_hidden'] = zeros.copy()
	generation['size_dnn_hidden'] = zeros.copy()
	generation['fitness'] = np.float32(zeros.copy()) - 1.0
	generation['batch_size'] = zeros.copy()
	generation['cross_prob'] = zeros.copy()
	generation['dnn_kl_weight'] = zeros.copy()
	generation['dnn_log_var_prior'] = zeros.copy()
	generation['dnn_weight'] = zeros.copy()
	generation['do_chckpt'] = np.bool8(zeros.copy())
	generation['hostname'] = ['127.0.0.1']*population_size
	generation['iterations'] = zeros.copy()
	generation['kl_anneal'] = zeros.copy()
	generation['log_dir'] = ['../data/logs']*population_size
	generation['model_dir'] = ['../data/models']*population_size
	generation['mutate_prob'] = zeros.copy()
	generation['num_epochs'] = zeros.copy()
	generation['optimizer'] = ['adam']*population_size
	generation['patience'] = zeros.copy()
	generation['population_size'] = zeros.copy()
	generation['prediction_log_var_prior'] = zeros.copy()
	generation['predictor_type'] = ['classification']*population_size
	generation['run_name'] = ['run_name']*population_size
	generation['table_dir'] = ['../data/tables']*population_size
	generation['time_stamp'] = zeros.copy()
	generation['train_file'] = ['train_file']*population_size
	generation['vae_kl_weight'] = zeros.copy()
	generation['vae_weight'] = zeros.copy()
	generation['w_kl_anneal'] = zeros.copy()

	generation['generationID'] = generation['generationID'] * generationID
	generation['generationID'] = np.int64(generation['generationID'])
	
	return generation

def generate_random_chromosomes(population_size,
						min_vae_hidden_layers = 1, max_vae_hidden_layers = 5, 
						min_dnn_hidden_layers = 1, max_dnn_hidden_layers = 5, 
						min_vae_hidden = 2, max_vae_hidden = 1024, 
						min_dnn_hidden = 2, max_dnn_hidden = 1024, 
						min_vae_latent = 2, max_vae_latent = 1024, 
						verbose=False):
	
	vae_nLayers_choices = range(min_vae_hidden_layers, max_vae_hidden_layers)
	dnn_nLayers_choices = range(min_dnn_hidden_layers, max_dnn_hidden_layers)
	vae_latent_choices = range(min_vae_latent, max_vae_latent)
	vae_nUnits_choices = range(min_vae_hidden, max_vae_hidden)
	dnn_nUnits_choices = range(min_dnn_hidden, max_dnn_hidden)
	
	generation = pd.DataFrame()
	
	zeros = np.zeros(population_size, dtype = int)
	ones = np.ones(population_size, dtype = int)
	arange = np.arange(population_size, dtype = int)

	generation['generationID'] = zeros.copy()
	generation['chromosomeID'] = arange.copy()
	generation['isTrained'] = zeros.copy()
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
	generation['fitness'] = np.float32(zeros.copy()) - 1.0

	# Place holders for after training
	generation['batch_size'] = zeros.copy()
	generation['cross_prob'] = zeros.copy()
	generation['dnn_kl_weight'] = zeros.copy()
	generation['dnn_log_var_prior'] = zeros.copy()
	generation['dnn_weight'] = zeros.copy()
	generation['do_chckpt'] = np.bool8(zeros.copy())
	generation['hostname'] = ['127.0.0.1']*population_size
	generation['iterations'] = zeros.copy()
	generation['kl_anneal'] = zeros.copy()
	generation['log_dir'] = ['../data/logs']*population_size
	generation['model_dir'] = ['../data/models']*population_size
	generation['mutate_prob'] = zeros.copy()
	generation['num_epochs'] = zeros.copy()
	generation['optimizer'] = ['adam']*population_size
	generation['patience'] = zeros.copy()
	generation['population_size'] = zeros.copy()
	generation['prediction_log_var_prior'] = zeros.copy()
	generation['predictor_type'] = ['classification']*population_size
	generation['run_name'] = ['run_name']*population_size
	generation['table_dir'] = ['../data/tables']*population_size
	generation['time_stamp'] = zeros.copy()
	generation['train_file'] = ['train_file']*population_size
	generation['vae_kl_weight'] = zeros.copy()
	generation['vae_weight'] = zeros.copy()
	generation['w_kl_anneal'] = zeros.copy()

	return generation

def get_machine(queue, bad_machines):
	machine = queue.get()
	sp_stdout_ = subprocess.STDOUT
	with open(os.devnull, 'wb') as devnull:
		callnow = "ping -c 1 {}".format(machine['host'])
		callnow = callnow.split(' ')
		
		try:
			check_ping = subprocess.check_call(callnow, 
						stdout=devnull, stderr=sp_stdout_)
		except Exception as error:
			check_ping = -1

		while check_ping != 0:
			print('Cannot reach host {}'.format(machine['host']))

			bad_machines.append(machine)
			assert(len(bad_machines) < queue.qsize()),\
				'Queue is empty while `bad_machines` is full'

			machine = queue.get()

			callnow = ("ping -c 1 " + machine['host']).split(' ')

			try:
				check_ping = subprocess.check_call(callnow, 
							stdout=devnull, stderr=sp_stdout_)
			except Exception as error:
				print(error)
				check_ping = -1

	return machine, bad_machines

def train_generation(generation, clargs, machines, private_key='id_ecdsa'):
	
	getChrom = 'https://LAUDeepGenerativeGenetics.pythonanywhere.com/GetChrom'
	key_filename = os.environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	generationID = generation.generationID.values[0]
	
	generation.generationID = np.int64(generation.generationID)
	generation.chromosomeID = np.int64(generation.chromosomeID)
	
	queue = mp.Queue()
	bad_machines = []
	
	#Create Processes
	for machine in machines: queue.put(machine)
	
	while True:
		# Run until entire Generation is listed as isTrained == True
		if all(generation.isTrained.values == 2): break
		
		for chromosome in generation.itertuples():
			chromosomeID = chromosome.chromosomeID
			
			if chromosome.isTrained == 0:# and queue.qsize() > 0:
				# Chromosome has never been touched
				info_message("\n\nCreating Process for Chromosome "\
						"{} on GenerationID {}".format(chromosome.chromosomeID,
							chromosome.generationID), end=" on machine ")
				
				# Find a Chromosome that is not trained yet
				# Wait for queue to have a value, 
				#	which is the ID of the machine that is done.
				machine, bad_machines = get_machine(queue, bad_machines)

				print('{}'.format(machine['host']))
				
				process = mp.Process(target=train_chromosome, 
									args=(chromosome, machine, queue, clargs))
				process.start()
				
				generation.set_value(chromosome.Index, 'isTrained', 1)
			
			if chromosome.isTrained != 2:
				# Check if chromosome has been updated on SQL
				
				sql_json = query_sql_database(chromosome.generationID, 
											  chromosome.chromosomeID, 
											  verbose = False)
				
				if isinstance(sql_json, requests.models.Response):
					warning_message('sql_json =?= sql_json.json()')
					sql_json = sql_json.json()

				elif isinstance(sql_json, dict):
					assert(sql_json['fitness'] > 0), \
						"[ERROR] If ID exists in SQL, why is fitness == -1?"\
						"\n GenerationID:{} ChromosomeID:{}".format(
							chromosome.generationID, chromosome.chromosomeID)
					
					for key, val in sql_json.items(): 
						debug_message('for key, val in sql_json.items():{}'.format(key))
						generation.set_value(chromosome.Index, key, val)
					
					generation.set_value(chromosome.Index, 'isTrained', 2)
				else:
					warning_message('SQL_JSON:{}'.format(sql_json))

			for bad_machine in bad_machines:
				# This lets us check if it is "good" again
				queue.put(bad_machine)

	for chromosome in generation.itertuples():
		assert(chromosome.isTrained), 'while loop should not have closed!'
		
		chromosomeID = chromosome.chromosomeID
		generationID = chromosome.generationID
		
		sql_json = query_sql_database(generationID, chromosomeID, 
										clargs=clargs, verbose=True)
		
		if sql_json['fitness'] is -1:
			sql_json = query_local_csv(generationID, chromosomeID, 
										clargs = clargs)
		
		if isinstance(sql_json, dict) and 'fitness' in sql_json.keys():
			assert(sql_json['fitness'] != -1), 'while loop may have failed!'
			
			if 'isTrained' not in sql_json.keys(): sql_json['isTrained'] = 2

			print('\n\n[INFO]')
			print('GenerationID:{}'.format(generationID))
			print('ChromosomeID:{}'.format(chromosomeID))
			print('Fitness:{}'.format(chromosome.fitness))
			print('Num VAE Layers:{}'.format(chromosome.num_vae_layers))
			print('Num DNN Layers:{}'.format(chromosome.num_dnn_layers))
			print('Size VAE Latent:{}'.format(chromosome.size_vae_latent))
			print('Size VAE Hidden:{}'.format(chromosome.size_vae_hidden))
			print('Size DNN Hidden:{}'.format(chromosome.size_dnn_hidden))
			print('\n\n')

			for col in generation.columns:
				generation.set_value(chromosomeID, col, sql_json[col])

	# After all is done: return what you received
	
	return generation

def generate_ssh_command(clargs, chromosome):
	command = []
	command.append('cd ~/vaelstmpredictor/examples; ')
	command.append('~/anaconda3/envs/tf_env/bin/python run_chromosome.py ')
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
	# command.append('--do_ckpt {}'.format(int(clargs.do_ckpt)))
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
				gituser = 'exowanderer', branchname = 'conv1d_model',
				port = 22, verbose = True, private_key='id_ecdsa'):
	
	key_filename = os.environ['HOME'] + '/.ssh/{}'.format(private_key)

	try:
		ssh = SSHClient()
		ssh.set_missing_host_key_policy(AutoAddPolicy())
		ssh.connect(hostname, key_filename = key_filename)
	except NoValidConnectionsError as error:
		warning_message(error)
		ssh.close()
		return
	
	command = []
	command.append('git clone https://github.com/{}/{}'.format(gituser,gitdir))
	command.append('cd {}'.format(gitdir))
	command.append('git pull')
	command.append('git checkout {}'.format(branchname))
	command.append('git pull')
	command = '; '.join(command)

	info_message('Executing {} on {}'.format(command, hostname))
	try:
		stdin, stdout, stderr = ssh.exec_command(command)
	except NoValidConnectionsError as error:
		warning_message(error)
		ssh.close()
		return

	info_message('Printing `stdout`')
	print_ssh_output(stdout)
	info_message('Printing `stderr`')
	print_ssh_output(stderr)
	
	ssh.close()
	info_message('SSH Closed on Git Clone')
	print("Git Clone Executed Successfully")


def print_ssh_output(ssh_output):
	debug_message('FLUSHING SSH_OUTPUT')
	print(ssh_output.readlines())
	# for line in ssh_output.readlines(): print(line)

def train_chromosome(chromosome, machine, queue, clargs, 
					port = 22, logdir = 'train_logs',
					git_dir = 'vaelstmpredictor',
					verbose = True):
	generationID = chromosome.generationID
	chromosomeID = chromosome.chromosomeID
	
	if not os.path.exists(logdir): os.mkdir(logdir)
	
	if verbose: 
		info_message('Checking if file {} exists on {}'.format(
										git_dir, machine['host']))
	
	# sys.stdout = open('{}/output{}.txt'.format(logdir, chromosomeID),'w')
	# sys.stderr = open('{}/error{}.txt'.format(logdir, chromosomeID), 'w')
	
	try:
		ssh = SSHClient()
		ssh.set_missing_host_key_policy(AutoAddPolicy())
		ssh.connect(machine["host"], key_filename=machine['key_filename'])
	except NoValidConnectionsError as error:
		warning_message(error)
		ssh.close()
		return
	
	stdin, stdout, stderr = ssh.exec_command('ls | grep {}'.format(git_dir))
	
	if(len(stdout.readlines()) == 0):
		git_clone()
	elif verbose: 
		
		info_message('File {} exists on {}'.format(git_dir, machine['host']))
	
	command = generate_ssh_command(clargs, chromosome)

	print("\n\nExecuting Train Chromosome Command:\n\t{}".format(command))
	
	stdin, stdout, stderr = ssh.exec_command(command)
	
	info_message('Printing `stdout` in Train Chromosome')
	print_ssh_output(stdout)
	info_message('Printing `stderr` in Train Chromosome')
	print_ssh_output(stderr)
	queue.put(machine)
	
	ssh.close()
	
	
	info_message('SSH Closed on Train Chromosome')
	info_message("Train Chromosome Executed Successfully: generationID:"\
					"{}\tchromosomeID:{}".format(generationID,chromosomeID))

def select_parents(generation):
	total_fitness = sum(chrom.fitness for chrom in generation.itertuples())
	#Generate two random numbers between 0 and total_fitness 
	#   not including total_fitness
	rand_parent1 = random.random()*total_fitness
	rand_parent2 = random.random()*total_fitness
	
	parent1 = None
	parent2 = None
	
	fitness_count = 0
	for chromosome in generation.itertuples():
		fitness_count += chromosome.fitness
		if(parent1 is None and fitness_count >= rand_parent1):
			parent1 = chromosome
		if(parent2 is None and fitness_count >= rand_parent2):
			parent2 = chromosome
		if(parent1 is not None and parent2 is not None):
			break

	return parent1, parent2

def cross_over(new_generation, generation, parent1, parent2, 
				chromosomeID, param_choices, prob, verbose=False):
	if verbose: info_message('Crossing over with probability: {}'.format(prob))

	idx_parent1 = parent1.Index
	idx_parent2 = parent2.Index

	if random.random() >= prob:
		crossover_happened = True
		for param in param_choices:
			p1_param = generation.iloc[idx_parent1][param]
			p2_param = generation.iloc[idx_parent2][param]
			child_gene = random.choice([p1_param, p2_param])
			new_generation.set_value(chromosomeID, param, child_gene)
	else: 
		crossover_happened = False
		
		p1_fitness = generation.iloc[idx_parent1]['fitness']
		p2_fitness = generation.iloc[idx_parent2]['fitness']

		idx_child = idx_parent1 if p1_fitness > p2_fitness else idx_parent1
		new_generation.iloc[chromosomeID] = generation.iloc[idx_child].copy()

	return new_generation, crossover_happened

def mutate(new_generation, generation, chromosomeID, 
			prob, param_choices, verbose = False):
	
	# explicit declaration
	zero = 0 

	if verbose:
		print('Mutating Child {} in Generation {}'.format(
			generation.loc[chromosomeID, 'chromosomeID'], 
			generation.loc[chromosomeID, 'generationID']))
		print('Mutating Child {} in Generation {}'.format(
			new_generation.loc[chromosomeID, 'chromosomeID'], 
			new_generation.loc[chromosomeID, 'generationID']))
	
	mutation_happened = False
	for param, (range_change, min_val) in param_choices.items():
		if(random.random() <= prob):
			mutation_happened = True

			# Compute delta_param step
			change_p = np.random.uniform(-range_change, range_change)

			# Add delta_param to param
			current_p = generation.loc[chromosomeID, param] + change_p
			
			# If param less than `min_val`, then set param to `min_val`
			current_p = np.max([current_p, min_val])
			current_p = np.int(np.round(current_p))

			# All params must be integer sized: round and convert
			new_generation.set_value(chromosomeID, param, current_p)

	return new_generation, mutation_happened