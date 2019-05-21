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
from time import time, sleep
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

def query_full_sql(loop_until_done=False, 
					hostname ='LAUDeepGenerativeGenetics.pythonanywhere.com', 
					sqlport=5000, sleep_time = 1):
	
	# getDatabase = 'http://{}:{}/getDatabase'.format(hostname, sqlport)
	# hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com'
	GetDatabase = 'http://{}/GetDatabase'.format(hostname)

	while True: # maybe use `for _ in range(iterations)` instead?
		req = requests.get(GetDatabase)
		try:
			sql_full_json = pd.DataFrame(req.json())
			
			# Toggle triggers if request+dataframe are successful
			return sql_full_json
		except Exception as error:
			message = '`query_full_sql` failed with error:\n{}'.format(error)
			warning_message(message)

		# Only triggers if requests+dataframe fails and not `loop_until_done`
		if not loop_until_done: return None
		sleep(sleep_time)

def query_generation(generationID, loop_until_done=False, 
		hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com', 
		# hostname ='172.16.50.176', 
		sqlport=5000, sleep_time = 1):
	
	# could add time_stamp,  to args and RESTful API call
	getGeneration = 'http://{}/GetGeneration'.format(hostname, sqlport)
	
	while True: # maybe use `for _ in range(iterations)` instead?
		json_ID = {'generationID':generationID}
		req = requests.get(getGeneration, params=json_ID)
		try:
			sql_generation = pd.DataFrame(req.json())
			
			# Toggle triggers if request+dataframe are successful
			# FINDME: Should probably sort `sql_generation` by `chromosomeID`
			return sql_generation
		except Exception as error:
			message = '`query_generation` failed with error:\n{}'.format(error)
			warning_message(message)

		# Only triggers if requests+dataframe fails and not `loop_until_done`
		if not loop_until_done: return None
		sleep(sleep_time)

def query_chromosome(generationID, chromosomeID, verbose=True,
					hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com', 
					sqlport = 5000):
	# getChromosome = 'http://{}:{}/GetChromosome'.format(hostname, sqlport)
	# hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com'
	getChromosome = 'http://{}/GetChromosome'.format(hostname)
	
	json_ID = {'generationID':generationID, 'chromosomeID':chromosomeID}
	
	sql_json = requests.get(getChromosome, params=json_ID)
	
	try:
		sql_json = sql_json.json()
	except Exception as error:
		warning_message('GeneticAlgorithm.py+query_chromosome+Exception:'
							'\n{}'.format(error))
	
	if sql_json == 0:# not isinstance(sql_json, requests.models.Response):
		if verbose: 
			warning_message('SQL Request Failed: sql_json = {} with {}'.format(
											sql_json, json_ID))
		return sql_json
	
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

def save_sql_to_csv(clargs):
	import pandas as pd
	import requests

	# hostname = clargs.hostname
	# sqlport = clargs.sqlport
	# getDatabase = 'http://{}:{}/GetDatabase'.format(hostname, sqlport)
	# hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com'
	GetDatabase = 'http://{}/GetDatabase'.format(clargs.sql_host)
	
	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(clargs.table_dir, 
									clargs.run_name, 
									clargs.time_stamp)


	req = requests.get(GetDatabase)
	sql_table = pd.DataFrame(req.json())
	sql_table.to_csv(table_name)

def create_blank_dataframe(generationID, population_size):

	generation = pd.DataFrame()
	
	zeros = np.zeros(population_size, dtype = int)
	ones = np.ones(population_size, dtype = int)
	arange = np.arange(population_size, dtype = int)

	generation['generationID'] = zeros + generationID
	generation['chromosomeID'] = arange
	generation['isTrained'] = zeros
	generation['num_vae_layers'] = zeros
	generation['num_dnn_layers'] = zeros
	generation['size_vae_latent'] = zeros
	generation['size_vae_hidden'] = zeros
	generation['size_dnn_hidden'] = zeros
	generation['fitness'] = np.float32(zeros) - 1.0
	generation['batch_size'] = zeros
	generation['cross_prob'] = zeros
	generation['dnn_kl_weight'] = zeros
	generation['dnn_log_var_prior'] = zeros
	generation['dnn_weight'] = zeros
	generation['do_chckpt'] = np.bool8(zeros)
	generation['hostname'] = ['127.0.0.1']*population_size
	generation['iterations'] = zeros
	generation['kl_anneal'] = zeros
	generation['log_dir'] = ['../data/logs']*population_size
	generation['model_dir'] = ['../data/models']*population_size
	generation['mutate_prob'] = zeros
	generation['num_epochs'] = zeros
	generation['optimizer'] = ['adam']*population_size
	generation['patience'] = zeros
	generation['population_size'] = zeros
	generation['prediction_log_var_prior'] = zeros
	generation['predictor_type'] = ['classification']*population_size
	generation['run_name'] = ['run_name']*population_size
	generation['table_dir'] = ['../data/tables']*population_size
	generation['time_stamp'] = zeros
	generation['train_file'] = ['train_file']*population_size
	generation['vae_kl_weight'] = zeros
	generation['vae_weight'] = zeros
	generation['w_kl_anneal'] = zeros

	return generation

def generate_random_chromosomes(population_size, geneationID = 0,
						min_vae_hidden_layers = 1, max_vae_hidden_layers = 5, 
						min_dnn_hidden_layers = 1, max_dnn_hidden_layers = 5, 
						min_vae_hidden = 2, max_vae_hidden = 1024, 
						min_dnn_hidden = 2, max_dnn_hidden = 1024, 
						min_vae_latent = 2, max_vae_latent = 1024, 
						verbose=False):
	
	# create blank dataframe with full SQL database required entrie
	generation = create_blank_dataframe(geneationID, population_size)
	
	# Overwrite chromosome parameters to evolve with random choices
	vae_nLayers_choices = range(min_vae_hidden_layers, max_vae_hidden_layers)
	dnn_nLayers_choices = range(min_dnn_hidden_layers, max_dnn_hidden_layers)
	vae_latent_choices = range(min_vae_latent, max_vae_latent)
	vae_nUnits_choices = range(min_vae_hidden, max_vae_hidden)
	dnn_nUnits_choices = range(min_dnn_hidden, max_dnn_hidden)
	
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
	
	return generation

def get_machine(queue):
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
			warning_message('Cannot reach host {}'.format(machine['host']))

			machine = queue.get()

			callnow = ("ping -c 1 " + machine['host']).split(' ')

			try:
				check_ping = subprocess.check_call(callnow, 
							stdout=devnull, stderr=sp_stdout_)
			except Exception as error:
				warning_message(error)
				check_ping = -1

	return machine

def process_generation(generation, queue, clargs):
	for chromosome in generation.itertuples():
		''' Chromosome has never been touched '''
		if chromosome.isTrained == 0:# and queue.qsize() >= 0:
			machine = get_machine(queue)

			info_message("Creating Process for Chromosome "\
						"{} on GenerationID {} on machine {}".format(
												chromosome.chromosomeID,
												chromosome.generationID,
												machine['host']))
			
			# Find a Chromosome that is not trained yet
			# Wait for queue to have a value, 
			#	which is the ID of the machine that is done.
			process = mp.Process(target=train_chromosome, 
								args=(chromosome, machine, queue, clargs))
			process.start()
			
			generation.set_value(chromosome.Index, 'isTrained', 1)
		""" # taken care of out of this function
		''' Check if chromosome has been updated on SQL '''
		if chromosome.isTrained != 2:
			sql_json = query_chromosome(chromosome.generationID, 
										  chromosome.chromosomeID, 
										  verbose = False)
			
			if isinstance(sql_json, requests.models.Response):
				warning_message('sql_json =?= sql_json.json()')
				try:
					sql_json = sql_json.json()
				except Exception as error:
					message = '`req.json()` Failed with:\n{}'.format(error)
					warning_message(message)

			if isinstance(sql_json, dict):
				assert(sql_json['fitness'] >= 0), \
					"[ERROR] If ID exists in SQL, why is fitness == -1?"\
					"\n GenerationID:{} ChromosomeID:{}".format(
						chromosome.generationID, chromosome.chromosomeID)
				
				info_message('Found Generation {}, Chromosome {}'.format(
						chromosome.generationID,chromosome.chromosomeID))
				
				for key, val in sql_json.items():
					generation.set_value(chromosome.Index, key, val)
				
				# generation.set_value(chromosome.Index, 'isTrained', 2)
		"""
	return generation

def train_generation(generation, clargs, machines, private_key='id_ecdsa', 
						verbose = False, sleep_time = 1):
	
	key_filename = os.environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	# Store `generationID` for easier use later
	generationID = generation.generationID.values[0]
	
	queue = mp.Queue()
	
	#Create Processes
	for machine in machines: queue.put(machine)
	
	count_while = 0
	start = time()

	# Start master process
	while not all(generation.isTrained.values == 2):
		generation = process_generation(generation, queue, clargs)
		while True:
			try:
				info_message('Querying Generation {} from SQL'.format(
									generationID))
				sql_generation = query_generation(generationID, 
												loop_until_done=False,
												sleep_time = sleep_time)
				break
			except Exception as error:
				message= 'tg1+query_generation failed because:{}'.format(error)
				warning_message(message)
				sleep(sleep_time)
		
		# If SQL does not exist yet or is not reachable, then keep processing
		if sql_generation is None: debug_message('sql_generation is None')
		if sql_generation is None: continue
		
		# Set all `isTrained==2` to `isTrained==0`
		for chromosome in generation.itertuples():
			if chromosome.isTrained == 2:
				generation.set_value(chromosome.chromosomeID,'isTrained', 0)

		# If chromosomeID exists in SQL and fitness >= 0, 
		#	then set `isTrained` back to 2 (i.e. "fully trained")
		#	This skips any entries with `isTrained == 1`
		for chromosome in sql_generation.itertuples():
			query = 'chromosomeID == {}'.format(chromosome.chromosomeID)
			
			# DEBUG: Alternative?
			# if sql_generation.query(query)['fitness'] >= 0:
			if sql_generation.at[chromosome.Index, 'fitness'] >= 0:
				generation.set_value(chromosome.chromosomeID, 'isTrained', 2)
			else:
				message = "sql_generation.at[chromosome.Index, 'fitness'] < 0"
				warning_message(message)
		
	'''After all chromosomes have been trained and stored in the SQL database,
		Download full generatin and copy data from SQL to local `generations`
	'''
	while True:
		try:
			info_message('Querying Generation {} from SQL'.format(
								generationID))
			sql_generation = query_generation(generationID, 
											loop_until_done=True,
											sleep_time = sleep_time)
			break
		except Exception as error:
			message= 'tg2+query_generation failed because:{}'.format(error)
			warning_message(message)
			sleep(sleep_time)
	
	assert(isinstance(sql_generation, pd.DataFrame)), \
			'`sql_generation` must be a dict'
	
	# assert(all(sql_generation.isTrained == 2)), \
	# 'while loop should not have closed!'

	# Assign sql data to generation dataframe
	# 	effectively: generation = sql_generation
	for chromosome in sql_generation.itertuples():
		for colname in sql_generation.columns:
			val = sql_generation.at[chromosome.Index, colname]
			generation.set_value(chromosome.Index, colname, val)

		if verbose:
			info_message('\n')
			print('GenerationID:{}'.format(chromosome.generationID))
			print('ChromosomeID:{}'.format(chromosome.chromosomeID))
			print('fitness:{}'.format(chromosome.fitness))
			print('Num VAE Layers:{}'.format(chromosome.num_vae_layers))
			print('Num DNN Layers:{}'.format(chromosome.num_dnn_layers))
			print('Size VAE Latent:{}'.format(chromosome.size_vae_latent))
			print('Size VAE Hidden:{}'.format(chromosome.size_vae_hidden))
			print('Size DNN Hidden:{}'.format(chromosome.size_dnn_hidden))
			print('\n\n')
	
	# After all is done: report back
	
	return generation#.astype(sql_generation.dtypes)

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
	command.append('--patience {}'.format(clargs.patience))
	command.append('--kl_anneal {}'.format(clargs.kl_anneal))
	command.append('--w_kl_anneal {}'.format(clargs.w_kl_anneal))
	command.append('--dnn_log_var_prior {}'.format(clargs.dnn_log_var_prior))
	command.append('--log_dir {}'.format(clargs.log_dir))
	command.append('--model_dir {}'.format(clargs.model_dir))
	command.append('--table_dir {}'.format(clargs.table_dir))
	command.append('--train_file {}'.format(clargs.train_file))
	command.append('--time_stamp {}'.format(int(clargs.time_stamp)))
	command.append('--hostname {}'.format(clargs.hostname))
	command.append('--sshport {}'.format(clargs.sshport))
	command.append('--num_vae_layers {}'.format(chromosome.num_vae_layers))
	command.append('--num_dnn_layers {}'.format(chromosome.num_dnn_layers))
	command.append('--size_vae_latent {}'.format(chromosome.size_vae_latent))
	command.append('--size_vae_hidden {}'.format(chromosome.size_vae_hidden))
	command.append('--size_dnn_hidden {}'.format(chromosome.size_dnn_hidden))
	command.append('--generationID {} '.format(chromosome.generationID))
	command.append('--chromosomeID {} '.format(chromosome.chromosomeID))

	# Boolean command line arguments
	if clargs.do_log: command.append('--do_log')
	if clargs.do_ckpt: command.append('--do_ckpt')
	if clargs.verbose: command.append('--verbose')
	if clargs.send_back: command.append('--send_back')
	if clargs.save_model: command.append('--save_model')
	
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

	info_message('Printing `stdout` in Git Clone')
	for line in stdout.readlines(): print(line)

	info_message('Printing `stderr` in Git Clone')
	for line in stderr.readlines(): print(line)
	
	ssh.close()
	info_message('SSH Closed on Git Clone')
	info_message("Git Clone Executed Successfully")

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

	info_message("\n\nExecuting Train Chromosome Command:\n\t{}".format(
																	command))
	
	stdin, stdout, stderr = ssh.exec_command(command)
	
	info_message('Printing `stdout` in Train Chromosome on '
					'{}'.format(machine['host']))
	
	for line in stdout.readlines(): print(line)

	info_message('Printing `stderr` in Train Chromosome'
					'{}'.format(machine['host']))

	for line in stderr.readlines(): print(line)

	queue.put(machine)
	
	ssh.close()
		
	info_message('SSH Closed on Train Chromosome')
	info_message("Train Chromosome Executed Successfully: generationID:"\
					"{}\tchromosomeID:{}".format(generationID,chromosomeID))

def select_parents(generation):
	'''Generate two random numbers between 0 and total_fitness 
		not including total_fitness'''

	total_fitness = generation.fitness.sum()
	assert(total_fitness >= 0), '`total_fitness` should not be negative'
	
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
	
	assert(None not in [parent1, parent2]),\
		'parent1 and parent2 must not be None:'\
		'Currently parent1:{}\tparent2:{}'.format(parent1, parent2)

	return parent1, parent2

def cross_over(new_generation, generation, parent1, parent2, 
				chromosomeID, param_choices, cross_prob, verbose=False):
	if verbose: info_message('Crossing over with probability: {}'.format(cross_prob))
	
	idx_parent1 = parent1.Index
	idx_parent2 = parent2.Index
	
	if random.random() <= cross_prob:
		crossover_happened = True
		
		for param in param_choices:
			p1_param = generation.loc[idx_parent1, param]
			p2_param = generation.loc[idx_parent2, param]
			
			child_gene = random.choice([p1_param, p2_param])
			new_generation.set_value(chromosomeID, param, child_gene)
	else: 
		crossover_happened = False
		
		p1_fitness = generation.loc[idx_parent1, 'fitness']
		p2_fitness = generation.loc[idx_parent2, 'fitness']
		
		idx_child = idx_parent1 if p1_fitness > p2_fitness else idx_parent1
		new_generation.iloc[chromosomeID] = generation.iloc[idx_child].copy()
	
	return new_generation.astype(generation.dtypes), crossover_happened

def mutate(new_generation, generation, chromosomeID, 
			mutate_prob, param_choices, verbose = False):
	
	# explicit declaration
	zero = 0 

	if verbose:
		info_message('Mutating Child {} in Generation {}'.format(
			generation.at[chromosomeID, 'chromosomeID'], 
			generation.at[chromosomeID, 'generationID']))
		info_message('Mutating Child {} in Generation {}'.format(
			new_generation.at[chromosomeID, 'chromosomeID'], 
			new_generation.at[chromosomeID, 'generationID']))
	
	mutation_happened = False
	for param, (range_change, min_val) in param_choices.items():
		if(random.random() <= mutate_prob):
			mutation_happened = True

			# Compute delta_param step
			change_p = np.random.uniform(-range_change, range_change)

			# Add delta_param to param
			current_p = generation.at[chromosomeID, param] + change_p
			
			# If param less than `min_val`, then set param to `min_val`
			current_p = np.max([current_p, min_val])
			current_p = np.int(np.round(current_p))

			# All params must be integer sized: round and convert
			new_generation.set_value(chromosomeID, param, current_p)

	return new_generation, mutation_happened