from warnings import simplefilter
simplefilter('ignore', category=DeprecationWarning)

import argparse
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import socket
import tensorflow as tf

from vaelstmpredictor.vae_conv1d_predictor.GeneticAlgorithm import *

from time import time, sleep
from vaelstmpredictor.utils.data_utils import MNISTData

import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	from paramiko import SSHClient, SFTPClient, Transport
	from paramiko import AutoAddPolicy, ECDSAKey
	from paramiko.ssh_exception import NoValidConnectionsError

warnings.filterwarnings(action='ignore',module='.*paramiko.*')

def debug_message(message): print('[DEBUG] {}'.format(message))
def warning_message(message): print('[WARNING] {}'.format(message))
def info_message(message): print('[INFO] {}'.format(message))

def make_sql_output(clargs, chromosome):
	output = {}
	output['run_name'] = clargs.run_name
	output['batch_size'] = clargs.batch_size
	output['cross_prob'] = clargs.cross_prob
	# output['do_ckpt'] = clargs.do_ckpt
	# output['do_log'] = clargs.do_log
	# output['send_back'] = clargs.send_back
	output['hostname'] = clargs.hostname
	output['iterations'] = clargs.num_generations
	output['kl_anneal'] = clargs.kl_anneal
	output['log_dir'] = clargs.log_dir
	output['model_dir'] = clargs.model_dir
	output['mutate_prob'] = clargs.mutate_prob
	output['num_epochs'] = clargs.num_epochs
	output['optimizer'] = clargs.optimizer
	output['patience'] = clargs.patience
	output['population_size'] = clargs.population_size
	output['prediction_log_var_prior'] = clargs.prediction_log_var_prior
	output['predictor_type'] = clargs.predictor_type
	output['table_dir'] = clargs.table_dir
	output['time_stamp'] = clargs.time_stamp
	output['train_file'] = clargs.train_file
	output['dnn_log_var_prior'] = clargs.dnn_log_var_prior
	output['dnn_kl_weight'] = clargs.dnn_kl_weight
	output['dnn_weight'] = clargs.dnn_weight
	output['vae_kl_weight'] = clargs.vae_kl_weight
	output['vae_weight'] = clargs.vae_weight
	# output['verbose'] = clargs.verbose
	output['w_kl_anneal'] = clargs.w_kl_anneal
	output['num_dnn_layers'] = clargs.num_dnn_layers
	output['num_vae_layers'] = clargs.num_vae_layers
	output['dnn_filter_size'] = clargs.dnn_filter_size
	output['vae_filter_size'] = clargs.vae_filter_size
	output['vae_latent_dim'] = clargs.vae_latent_dim
	output['generationID'] = chromosome.generationID
	output['chromosomeID'] = chromosome.chromosomeID
	output['isTrained'] = chromosome.isTrained
	output['fitness'] = chromosome.fitness

	return output

def sftp_send(local_file, remote_file, hostname, port, key_filename, 
				verbose = False):
	
	transport = Transport((clargs.hostname, clargs.sshport))
	pk = ECDSAKey.from_private_key(open(key_filename))
	transport.connect(username = 'acc', pkey=pk)

	if chromosome.verbose: 
		info_message('SFTPing Model Weights from {} to {} on {}'.format(
										local_file, remote_file, hostname))
	
	sftp = SFTPClient.from_transport(transport)
	sftp.put(local_file, remote_file)

	sftp.close()
	transport.close()

def ssh_command(command, clargs, print_output = False):

	private_key = os.environ['HOME'] + '/.ssh/{}'.format('id_ecdsa')
	
	ssh = SSHClient()
	ssh.set_missing_host_key_policy(AutoAddPolicy())
	ssh.connect(clargs.hostname, key_filename=private_key)
	stdin, stdout, stderr = ssh.exec_command(command)
	
	if print_output:
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

	ssh.close()
	
	return stdin, stdout, stderr

def ssh_out_table_entry(clargs, chromosome):
	table_dir = clargs.table_dir
	table_name = '{}/{}_fitness_table_{}.csv'
	table_name = table_name.format(clargs.table_dir, 
									clargs.run_name, 
									clargs.time_stamp)

	remote_table_name = table_name.replace('../','')

	check_table_exists = 'ls {}'.format(remote_table_name)
	stdin, stdout, stderr = ssh_command(check_table_exists, clargs)
	
	stdout.channel.recv_exit_status()
	if(len(stdout.readlines()) == 0):
		# Check if the file exists, with proper header
		touch_command = 'touch {}'.format(remote_table_name)
		
		header = []
		header.append('generationID'.format(clargs.generationID))
		header.append('chromosomeID'.format(clargs.chromosomeID))
		header.append('fitness'.format(chromosome.fitness))
		header.extend(list(clargs.__dict__.keys()))
		header = ','.join(header)

		create_header = "echo {} >> {}".format(header, remote_table_name)

		info_message('Creating {} with header {}'.format(
									remote_table_name, create_header))

		stdin, stdout, stderr = ssh_command(create_header, clargs)
	elif verbose: 
			info_message('File {} exists on {}'.format(
								remote_table_name, machine['host']))

	entry = []
	entry.append('{}'.format(clargs.generationID))
	entry.append('{}'.format(clargs.chromosomeID))
	entry.append('{}'.format(chromosome.fitness))
	
	for key,val in clargs.__dict__.items():
		if key not in ['generationID', 'chromosomeID']:
			entry.append('{}'.format(val))

	entry = ','.join(entry)

	command = ['echo "{}"'.format(entry)]
	command.append('>> vaelstmpredictor/{}'.format(remote_table_name))
	command = ' '.join(command)
	
	if chromosome.verbose:
		info_message('\n')
		print('Remotely entry storing ON {}'.format(clargs.hostname))
		print('\n\n{}'.format(entry))
		print('\n\nAT vaelstmpredictor/{} '.format(remote_table_name))
		print('\n\nWITH \n\n{}'.format(command))

	stdin, stdout, stderr = ssh_command(command, clargs, print_output = False)

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
	parser.add_argument('--num_epochs', type=int, default=1,
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
	parser.add_argument("--do_log", action="store_true", 
			help="save log files")
	parser.add_argument("--do_ckpt", action="store_true",
			help="save model checkpoints")
	parser.add_argument('--patience', type=int, default=10,
			help='# of epochs, for early stopping')
	parser.add_argument("--kl_anneal", type=int, default=0, 
			help="number of epochs before kl loss term is 1.0")
	parser.add_argument("--w_kl_anneal", type=int, default=0, 
			help="number of epochs before w's kl loss term is 1.0")
	parser.add_argument('--dnn_log_var_prior', type=float, default=0.0,
			help='Prior on the log variance for the DNN predictor')
	parser.add_argument('--log_dir', type=str, default='../data/logs',
			help='basedir for saving log files')
	parser.add_argument('--model_dir', type=str, default='../data/models',
			help='basedir for saving model weights')
	parser.add_argument('--table_dir', type=str, default='../data/tables',
			help='basedir for storing the table of params and fitnesses.')
	parser.add_argument('--train_file', type=str, default='MNIST',
			help='file of training data (.pickle)')
	parser.add_argument('--time_stamp', type=int, default=0,
			help='Keeps track of runs and re-runs')
	parser.add_argument('--sql_host', type=str, 
			default='LAUDeepGenerativeGenetics.pythonanywhere.com',
			help='SQL Server Location')
	parser.add_argument('--hostname', type=str, default='127.0.0.1',
			help='The hostname of the computer to send results back to.')
	parser.add_argument('--sshport', type=int, default=22,
			help='The port on the work computer to send ssh over.')
	parser.add_argument('--sqlport', type=int, default=5000,
			help='The port on the work computer to send ssh over.')
	parser.add_argument('--send_back', action='store_true', 
			help='Toggle whether to send the ckpt file + population local csv')
	parser.add_argument('--generationID', type=int, default=-1,
			help='Chromosome Generation ID')
	parser.add_argument('--chromosomeID', type=int, default=-1,
			help='Chromosome Chromosome ID')
	parser.add_argument('--num_vae_layers', type=int, default=1,
			help='Depth of the VAE')
	parser.add_argument('--num_dnn_layers', type=int, default=1,
			help='Depth of the DNN')
	parser.add_argument('--vae_latent_dim', type=int, default=16,
			help='Size of the VAE Latent Layer')
	parser.add_argument('--vae_filter_size', type=int, default=16,
			help='Size of the VAE Hidden Layer')
	parser.add_argument('--dnn_filter_size', type=int, default=16,
			help='Size of the DNN Hidden Layer')
	parser.add_argument('--save_model', action='store_true',
			help='Save model ckpt.s and other stored values')
	parser.add_argument('--verbose', action='store_true',
			help='print more [INFO] and [DEBUG] statements')
	parser.add_argument('--cross_prob', type=float, default = 0.7,
			help='Probability that crossover will occur: for SQL storage')
	parser.add_argument('--mutate_prob', type=float, default = 0.01,
			help='Probability that mutate will occur: for SQL storage')
	parser.add_argument('--num_generations', type=int, default = 25,
			help='Number of generations to process: for SQL storage')
	parser.add_argument('--population_size', type=float, default = 100,
			help='Number of members per generation: for SQL storage')
	
	clargs = parser.parse_args()
	
	# clargs.do_log = True
	# clargs.do_ckpt = True
	# clargs.verbose = True
	# clargs.cross_prob = 0.7
	# clargs.mutate_prob = 0.01
	# clargs.num_generations = 100
	# clargs.population_size = 100

	for key,val in clargs.__dict__.items(): 
		if 'dir' in key: 
			if not os.path.exists(val): 
				os.mkdir(val)

	chrom_params = {}
	chrom_params['verbose'] = clargs.verbose
	chrom_params['save_model'] = clargs.save_model
	chrom_params['num_vae_layers'] = clargs.num_vae_layers
	chrom_params['num_dnn_layers'] = clargs.num_dnn_layers
	chrom_params['dnn_filter_size'] = clargs.dnn_filter_size
	chrom_params['vae_filter_size'] = clargs.vae_filter_size
	chrom_params['dnn_filter_size'] = clargs.dnn_filter_size
	chrom_params['vae_latent_dim'] = clargs.vae_latent_dim
	chrom_params['generationID'] = clargs.generationID
	chrom_params['chromosomeID'] = clargs.chromosomeID
	chrom_params['vae_weight'] = clargs.vae_weight
	chrom_params['vae_kl_weight'] = clargs.vae_kl_weight
	chrom_params['dnn_weight'] = clargs.dnn_weight
	chrom_params['dnn_kl_weight'] = clargs.dnn_kl_weight
	chrom_params['clargs'] = clargs
	
	data_instance = MNISTData(batch_size = clargs.batch_size)
	
	n_train, n_features = data_instance.data_train.shape
	n_test, n_features = data_instance.data_valid.shape

	clargs.original_dim = n_features
	clargs.n_labels = len(np.unique(data_instance.train_labels))
	
	chrom_params['data_instance'] = data_instance

	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	hostname = s.getsockname()[0]
	s.close()

	info_message('\n\nParams for this VAE_NN:')
	for key,val in clargs.__dict__.items():
		print('{:20}{}'.format(key,val))

	chromosome = Chromosome(**chrom_params)
	chromosome.verbose = True
	chromosome.train(verbose = True)
	
	generationID = chromosome.generationID
	chromosomeID = chromosome.chromosomeID

	key_filename = os.environ['HOME'] + '/.ssh/{}'.format('id_ecdsa')

	local_wghts = chromosome.wghts_save_loc
	remote_wghts = 'vaelstmpredictor/{}'.format(local_wghts)
	remote_wghts = remote_wghts.replace('../','')
	
	if clargs.send_back:
		sftp_send(local_file = local_wghts, 
					remote_file = remote_wghts,
					hostname = clargs.hostname, 
					port = clargs.sshport, 
					key_filename = key_filename, 
					verbose = clargs.verbose)

		ssh_out_table_entry(clargs, chromosome)
	
	info_message('\n')
	print('Result: ', end=" ")
	print('GenerationID: {}'.format(chromosome.generationID), end=" ")
	print('ChromosomeID: {}'.format(chromosome.chromosomeID), end=" ")
	print('Fitness: {}\n'.format(chromosome.fitness))
	
	chromosome.isTrained = 2 # Set "is fully trained"
	put_sql_dict = make_sql_output(clargs, chromosome)

	output_table_name = '{}/{}_{}_{}_trained_model_entry_{}.save'
	output_table_name = output_table_name.format(clargs.table_dir, 
												clargs.run_name, 
												chromosome.generationID, 
												chromosome.chromosomeID, 
												clargs.time_stamp)
	
	joblib.dump(put_sql_dict, output_table_name)

	local_output_table = output_table_name
	remote_output_table = 'vaelstmpredictor/{}'.format(output_table_name)
	remote_output_table = remote_output_table.replace('../','')

	if clargs.send_back:
		sftp_send(local_file = local_output_table, 
					remote_file = remote_output_table,
					hostname = clargs.hostname, 
					port = clargs.sshport, 
					key_filename = key_filename, 
					verbose = clargs.verbose)
	
	# DEBUG: For some reason the RESTful API does not like these 3 pieces
	remove_question_marks = True
	if remove_question_marks:
		if 'send_back' in put_sql_dict.keys():
			del put_sql_dict['send_back']
		
		if 'do_log' in put_sql_dict.keys():
			del put_sql_dict['do_log']
		
		if 'verbose' in put_sql_dict.keys():
			del put_sql_dict['verbose']
	
	# hostname = 'LAUDeepGenerativeGenetics.pythonanywhere.com'
	AddChrom = 'http://{}/AddChrom'.format(clargs.sql_host)
	# AddChrom= 'http://{}:{}/AddChrom'.format(clargs.sql_host, clargs.sqlport)
	
	info_message('Storing to SQL db at {}'.format(AddChrom))
	
	req = requests.get(url = AddChrom, params = put_sql_dict)
	
	try:
		if req.json() == 1:
			info_message('Remote SQL Entry Added Successfully')
		else:
			warning_message('\n\n!! The World Has Ended !!\n\n')
	except json.decoder.JSONDecodeError as error: 
		warning_message(error)
