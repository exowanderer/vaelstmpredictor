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
	debug_message('1,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))
	getFitness = 'http://LAUDeepGenerativeGenetics.pythonanywhere.com/'
	getFitness = getFitness + 'GetFitness'
	debug_message('2,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))
	if clargs is not None:
		debug_message('3,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))
		table_dir = clargs.table_dir
		table_name = '{}/{}_{}_{}_sql_fitness_table_{}.json'
		table_name = table_name.format(clargs.table_dir, 
						clargs.run_name, generationID, 
						chromosomeID, clargs.time_stamp)
		debug_message('4,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))

	debug_message('5,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))

	json_ID = {'generationID':generationID, 'chromosomeID':chromosomeID}
	debug_message('6,query_sql_database+generationID:'
					'{}+chromosomeID:{}'.format(generationID, chromosomeID))
	sql_json = requests.get(getFitness, params=json_ID)
	debug_message('7,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
	debug_message('7b,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, sql_json))

	print(getFitness,end="?")
	for key,val in json_ID.items():
		print('{}={}&'.format(key,val), end="")
	print('')

	try:
		sql_json = sql_json.json()
	except Exception as error:
		warning_message('query_sql_database+Except:\n{}'.format(error))
	
	debug_message('7c,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, sql_json))
	
	if sql_json == 0:#not isinstance(sql_json, requests.models.Response):
		debug_message('8,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
		if verbose: 
			print('SQL Request Failed: sql_json = {} with {}'.format(sql_json, 
																	json_ID))
		debug_message('9,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
		
		return sql_json

	debug_message('10,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
	# Only triggered if `sql_json` is a `dict`
	sql_json = sql_json.json()
	debug_message('11,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))

	if clargs is not None:
		debug_message('12,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
		with open(table_name, 'a') as f_out: 
			json.dump(sql_json, f_out)

	debug_message('13,query_sql_database+generationID:'
					'{}+chromosomeID:{}+sql_json:{}'.format(
						generationID, chromosomeID, type(sql_json)))
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
	
	generation['generationID'] = np.ones(population_size, dtype = int)
	generation['chromosomeID'] = np.arange(population_size, dtype = int)
	generation['isTrained'] = np.zeros(population_size, dtype = int)
	generation['num_vae_layers'] = np.zeros(population_size, dtype = int)
	generation['num_dnn_layers'] = np.zeros(population_size, dtype = int)
	generation['size_vae_latent'] = np.zeros(population_size, dtype = int)
	generation['size_vae_hidden'] = np.zeros(population_size, dtype = int)
	generation['size_dnn_hidden'] = np.zeros(population_size, dtype = int)
	generation['fitness'] = np.zeros(population_size, dtype = float) - 1.0

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
	generation['generationID'] = np.zeros(population_size, dtype = int)
	generation['chromosomeID'] = np.arange(population_size, dtype = int)
	generation['isTrained'] = np.zeros(population_size, dtype = int)
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
	debug_message('1,tg+generationID:{}'.format(generation.generationID))
	getChrom = 'https://LAUDeepGenerativeGenetics.pythonanywhere.com/GetChrom'
	key_filename = os.environ['HOME'] + '/.ssh/{}'.format(private_key)
	
	generation.generationID = np.int64(generation.generationID)
	generation.chromosomeID = np.int64(generation.chromosomeID)
	
	queue = mp.Queue()
	bad_machines = []
	
	#Create Processes
	for machine in machines: queue.put(machine)
	debug_message('2,tg+generationID:{}'.format(generation.generationID))
	while True:
		generationID = generation.generationID.values[0]
		debug_message('2b,tg+while+generationID:{}+queue size:{}'.format(
							generationID, queue.qsize()))
		debug_message('3,tg+while+generationID:{}'.format(generationID))
		
		# Run until entire Generation is listed as isTrained == True
		if all(generation.isTrained.values == 2): break
		debug_message('4,tg+while+generationID:{}'.format(generationID))
		for chromosome in generation.itertuples():
			chromosomeID = chromosome.chromosomeID
			debug_message('5,tg+while+for+generationID:{}'.format(
				generationID, chromosomeID))
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
				debug_message('6,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))

				process = mp.Process(target=train_chromosome, 
									args=(chromosome, machine, queue, clargs))
				process.start()
				debug_message('7,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
				generation.set_value(chromosome.Index, 'isTrained', 1)
				debug_message('8,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
			debug_message('9,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
			if chromosome.isTrained != 2:
				# Check if chromosome has been updated on SQL
				debug_message('10,tg+while+for+generationID:'
							'{}+chromosomeID:{}+isTrained:{}'.format(
							generationID, chromosomeID,chromosome.isTrained))
				sql_json = query_sql_database(chromosome.generationID, 
											  chromosome.chromosomeID, 
											  verbose = False)
				debug_message('11,tg+while+for+generationID:'
							'{}+chromosomeID:{}+sql_json:{}'.format(
								generationID, chromosomeID,sql_json))
				
				if isinstance(sql_json, requests.models.Response):
					warning_message('sql_json =?= sql_json.json()')
					sql_json = sql_json.json()

				elif isinstance(sql_json, dict):
					debug_message('12,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
					assert(sql_json['fitness'] > 0), \
						"[ERROR] If ID exists in SQL, why is fitness == -1?"\
						"\n GenerationID:{} ChromosomeID:{}".format(
							chromosome.generationID, chromosome.chromosomeID)
					debug_message('13,tg+while+for+generationID:'
						'{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
					
					debug_message('14,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))

					for key, val in sql_json.items(): 
						generation.set_value(chromosome.Index, key, val)

					generation.set_value(chromosome.Index, 'isTrained', 2)

					debug_message('15,tg+while+for+generationID:'
							'{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
				else:
					warning_message('SQL_JSON:{}'.format(sql_json))

				debug_message('16,tg+while+for+generationID:'
							'generationID:{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
			
			debug_message('17,tg+while+for+generationID:'
							'generationID:{}+chromosomeID:{}'.format(
								generationID, chromosomeID))

			for bad_machine in bad_machines:
				# This lets us check if it is "good" again
				queue.put(bad_machine)

			debug_message('18,tg+while+for+generationID:'
							'generationID:{}+chromosomeID:{}'.format(
								generationID, chromosomeID))
	debug_message('19,tg+while+for+generationID:'
					'generationID:{}'.format(generationID))

	for chromosome in generation.itertuples():
		debug_message('20,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		assert(chromosome.isTrained), 'while loop should not have closed!'
		debug_message('21,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		chromosomeID = chromosome.chromosomeID
		generationID = chromosome.generationID
		debug_message('22,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		sql_json = query_sql_database(generationID, chromosomeID, 
										clargs=clargs, verbose=True)
		debug_message('23,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		if sql_json['fitness'] is -1:
			debug_message('24,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
			sql_json = query_local_csv(generationID, chromosomeID, 
										clargs = clargs)
		debug_message('25,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		if isinstance(sql_json, dict) and 'fitness' in sql_json.keys():
			assert(sql_json['fitness'] != -1), 'while loop may have failed!'
			debug_message('26,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
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

			debug_message('27,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))

			for col in generation.columns:
				generation.set_value(chromosomeID, col, sql_json[col])

			debug_message('28,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
		debug_message('29,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
	debug_message('30,tg+while+for+generationID:'
					'generationID:{}+chromosomeID:{}'.format(
						generationID, chromosomeID))
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
	# print_ssh_output(stdout)
	info_message('Printing `stderr`')
	# print_ssh_output(stderr)
	
	ssh.close()
	info_message('SSH Closed on Git Clone')
	print("Git Clone Executed Successfully")

'''
def  print_ssh_output(ssh_output):
	debug_message('INSIDE: # print_ssh_output')
	try:
		debug_message('INSIDE: TRY1')
		try:
			debug_message('INSIDE: TRY2')
			ssh_output.channel.recv_exit_status()
			debug_message('INSIDE: TRY2 DONE')
		except Exception as error:
			debug_message('INSIDE: EXCEPT2')
			warning_message('\n\n1,Error on ssh_output.readlines():'
						'{}'.format(error))
			debug_message('INSIDE: FINSIHED EXCEPT2')
		debug_message('INSIDE: Continuing TRY1')
		for line in ssh_output.readlines(): print(line)
		debug_message('INSIDE: FINSIHED TRY1')
	except Exception as error:
		debug_message('INSIDE: EXCEPT1')
		warning_message('\n\n2,Error on ssh_output.readlines():'
						'{}'.format(error))
		debug_message('INSIDE: FINSIHED EXCEPT1')
'''
def train_chromosome(chromosome, machine, queue, clargs, 
					port = 22, logdir = 'train_logs',
					git_dir = 'vaelstmpredictor',
					verbose = True):
	generationID = chromosome.generationID
	chromosomeID = chromosome.chromosomeID
	
	debug_message('1,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	
	if not os.path.exists(logdir): os.mkdir(logdir)
	debug_message('2,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	if verbose: 
		info_message('Checking if file {} exists on {}'.format(
										git_dir, machine['host']))
	debug_message('3,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	# sys.stdout = open('{}/output{}.txt'.format(logdir, chromosomeID),'w')
	# sys.stderr = open('{}/error{}.txt'.format(logdir, chromosomeID), 'w')
	
	try:
		debug_message('4,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
		ssh = SSHClient()
		ssh.set_missing_host_key_policy(AutoAddPolicy())
		ssh.connect(machine["host"], key_filename=machine['key_filename'])
		debug_message('5,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	except NoValidConnectionsError as error:
		debug_message('5b,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
		warning_message(error)
		ssh.close()
		return
	debug_message('6,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	stdin, stdout, stderr = ssh.exec_command('ls | grep {}'.format(git_dir))
	debug_message('7,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	if(len(stdout.readlines()) == 0):
		debug_message('8,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
		git_clone()
		debug_message('9,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	elif verbose: 
		debug_message('10,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
		info_message('File {} exists on {}'.format(git_dir, machine['host']))
		debug_message('11,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))

	debug_message('12,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	command = generate_ssh_command(clargs, chromosome)

	debug_message('13,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	
	print("\n\nExecuting Train Chromosome Command:\n\t{}".format(command))
	debug_message('14,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	stdin, stdout, stderr = ssh.exec_command(command)
	debug_message('15,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	
	info_message('Printing `stdout` in Train Chromosome')
	# print_ssh_output(stdout)
	info_message('Printing `stderr` in Train Chromosome')
	# print_ssh_output(stderr)
	debug_message('16,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	debug_message('queue size:{}'.format(queue.qsize()))
	queue.put(machine)
	debug_message('queue size:{}'.format(queue.qsize()))
	debug_message('17,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	ssh.close()
	
	debug_message('18,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))
	info_message('SSH Closed on Train Chromosome')
	info_message("Train Chromosome Executed Successfully: generationID:"\
					"{}\tchromosomeID:{}".format(generationID,chromosomeID))
	debug_message('19,tc+generationID:{}+chromosomeID:{}'.format(
					generationID, chromosomeID))

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
		print('Mutating Child {} in Generation {}'.format(child.chromosomeID, 
														 child.generationID))
	
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