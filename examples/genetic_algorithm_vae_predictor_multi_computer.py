# from https://github.com/exowanderer/vaelstmpredictor
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --num_generations 500 --population_size 10 --num_epochs 200
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import socket
from time import time
import json
import requests
import sys

from sklearn.externals import joblib
from contextlib import redirect_stdout

from keras import backend as K
from keras.utils import to_categorical
from tqdm import tqdm

from vaelstmpredictor.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmpredictor.utils.model_utils import save_model_in_pieces
from vaelstmpredictor.utils.model_utils import AnnealLossWeight
from vaelstmpredictor.utils.data_utils import MNISTData
from vaelstmpredictor.utils.weightnorm import data_based_init
from vaelstmpredictor.vae_predictor.dense_model import VAEPredictor
from vaelstmpredictor.vae_predictor.train import train_vae_predictor

from vaelstmpredictor.GeneticAlgorithm import *

def debuge_message(message): print('[DEBUG] {}'.format(message))
def info_message(message): print('[INFO] {}'.format(message))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_name', type=str, default='deleteme',
				help='tag for current run')
	parser.add_argument('--predictor_type', type=str, default="classification",
				help='select `classification` or `regression`')
	parser.add_argument('--batch_size', type=int, default=128,
				help='batch size')
	parser.add_argument('--optimizer', type=str, default='adam',
				help='optimizer name') 
	parser.add_argument('--num_epochs', type=int, default=1,
				help='number of epochs')
	parser.add_argument('--max_vae_hidden_layers', type=int, default=5, 
				 help='Maximum number of VAE hidden layers')
	parser.add_argument('--max_vae_latent', type=int, default=512, 
				 help='Maximum number of VAE neurons per layer')
	parser.add_argument('--max_dnn_latent', type=int, default=512, 
				 help='Maximum number of DNN neurons per layer')
	parser.add_argument('--max_vae_hidden', type=int, default=512, 
				 help='Maximum number of VAE neurons per layer')
	parser.add_argument('--max_dnn_hidden', type=int, default=512, 
				 help='Maximum number of VAE neurons per layer')
	parser.add_argument('--max_dnn_hidden_layers', type=int, default=5,
				 help='Maximum number of DNN hidden layers')
	parser.add_argument('--min_vae_hidden_layers', type=int, default=1, 
				 help='minimum number of VAE hidden layers')
	parser.add_argument('--min_vae_latent', type=int, default=2, 
				 help='minimum number of VAE neurons per layer')
	parser.add_argument('--min_dnn_latent', type=int, default=2, 
				 help='minimum number of DNN neurons per layer')
	parser.add_argument('--min_vae_hidden', type=int, default=2, 
				 help='Maximum number of VAE neurons per layer')
	parser.add_argument('--min_dnn_hidden', type=int, default=2, 
				 help='Maximum number of VAE neurons per layer')
	parser.add_argument('--min_dnn_hidden_layers', type=int, default=1,
				 help='minimum number of DNN hidden layers')	
	parser.add_argument('--dnn_weight', type=float, default=1.0,
				help='relative weight on prediction loss')
	parser.add_argument('--vae_weight', type=float, default=30.53,
				help='relative weight on prediction loss')
	parser.add_argument('--vae_kl_weight', type=float, default=1.39e6,
				help='relative weight on prediction loss')
	parser.add_argument('--dnn_kl_weight', type=float, default=6.35,
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
	parser.add_argument('--cross_prob', type=float, default=0.7,
				help='Probability of crossover between generations')
	parser.add_argument('--mutate_prob', type=float, default=0.01,
				help='Probability of mutation for each member')
	parser.add_argument('--population_size', type=int, default=3,
				help='size of the population to evolve; '\
						'preferably divisible by 2')
	parser.add_argument('--num_generations', type=int, default=2,
				help='number of generations for genetic algorithm')
	parser.add_argument('--verbose', action='store_true',
				help='print more [INFO] and [DEBUG] statements')
	parser.add_argument('--make_plots', action='store_true',
				help='make plots of growth in the best_loss over generations')
	parser.add_argument('--port', type=int, default=22,
				help='IP port over which to ssh')
	clargs = parser.parse_args()
	
	for key,val in clargs.__dict__.items(): 
		if 'dir' in key: 
			if not os.path.exists(val): 
				os.mkdir(val)

	key_filename = os.environ['HOME'] + '/.ssh/{}'.format('id_ecdsa')

	machines = [{"host": "172.16.50.181", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.176", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.177", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.163", "username": "acc", 
					"key_filename": key_filename},
				{"host": "172.16.50.182", "username": "acc", 
					"key_filename": key_filename},# not operation today
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

	
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	hostname = s.getsockname()[0]
	s.close()
	
	clargs.hostname = hostname
	
	run_name = clargs.run_name
	num_epochs = clargs.num_epochs
	cross_prob = clargs.cross_prob
	mutate_prob = clargs.mutate_prob
	population_size = clargs.population_size
	num_generations = clargs.num_generations
	verbose = clargs.verbose
	make_plots = clargs.make_plots
	
	clargs.data_type = 'MNIST'
	data_instance = MNISTData(batch_size = clargs.batch_size)
	
	n_train, n_features = data_instance.data_train.shape
	n_test, n_features = data_instance.data_valid.shape
	
	clargs.original_dim = n_features
	
	clargs.time_stamp = int(time())
	clargs.run_name = '{}_{}_{}'.format(clargs.run_name, 
								clargs.data_type, clargs.time_stamp)
	
	if verbose: print('\n\n[INFO] Run Base Name: {}\n'.format(clargs.run_name))
	
	clargs.n_labels = len(np.unique(data_instance.train_labels))
	
	generation = generate_random_chromosomes(population_size = population_size,
						min_vae_hidden_layers = clargs.min_vae_hidden_layers,
						min_dnn_hidden_layers = clargs.min_dnn_hidden_layers,
						max_vae_hidden_layers = clargs.max_vae_hidden_layers,
						max_dnn_hidden_layers = clargs.max_dnn_hidden_layers,
						min_vae_hidden = clargs.min_vae_hidden,
						max_vae_hidden = clargs.max_vae_hidden,
						min_dnn_hidden = clargs.min_dnn_hidden,
						max_dnn_hidden = clargs.max_dnn_hidden,
						min_vae_latent = clargs.min_vae_latent,
						max_vae_latent = clargs.max_vae_latent,
						verbose = clargs.verbose)
	
	generationID = 0
	debuge_message('\n\n')
	debuge_message('0,__main__+generationID:{}'.format(generationID))
	debuge_message('\n\n')
	generation = train_generation(generation, clargs, machines)
	debuge_message('\n\n')
	debuge_message('0b,__main__+generationID:{}'.format(generationID))
	debuge_message('\n\n')
	best_fitness = []
	fitnesses = [chromosome.fitness for _, chromosome in generation.iterrows()]
	
	new_best_fitness = max(fitnesses)
	
	if clargs.verbose:
		info_message('For Generation: {}, the best fitness was {}'.format(
				generationID, new_best_fitness))
	
	best_fitness.append(new_best_fitness)
	
	if make_plots:
		fig = plt.gcf()
		fig.show()
	
	param_choices = {'num_vae_layers': (1,1), 
					 'num_dnn_layers': (1,1), 
					 'size_vae_latent': (10,1), 
					 'size_vae_hidden': (50,1), 
					 'size_dnn_hidden': (50,1)}
	
	start = time()
	# while gen_num < num_generations:
	for generationID in range(1,num_generations):
		start_while = time()
		# Create new generation
		debuge_message('\n\n')
		debuge_message('1,__main__+loop+generationID:{}'.format(generationID))
		debuge_message('\n\n')
		new_generation = create_blank_dataframe(generationID, population_size)
		
		for chromosomeID in tqdm(range(population_size)):
			debuge_message('\n\n')
			debuge_message('2,__main__+loop2+generationID,'\
				'chromosomeID:{}'.format(generationID, chromosomeID))
			debuge_message('\n\n')
			parent1, parent2 = select_parents(generation)
			
			child, crossover_happened = cross_over(new_generation, generation,
											parent1, parent2, chromosomeID,
											param_choices.keys(), cross_prob, 
											verbose = verbose)
			
			new_generation, mutation_happened = mutate(
											new_generation, generation,
											chromosomeID, mutate_prob, 
											param_choices, verbose = verbose)
			
			isTrained = mutation_happened*crossover_happened
			if isTrained: isTrained = 2

			new_generation.set_value(chromosomeID, 'isTrained', isTrained)
			new_generation.set_value(chromosomeID, 'generationID',generationID)
			new_generation.set_value(chromosomeID, 'chromosomeID',chromosomeID)
			new_generation.set_value(chromosomeID, 'fitness', -1)
			
			info_message('Adding Chromosome:\n{}'.format(child))
			# new_generation.iloc[chromosomeID] = child
		
		# Re-sort by chromosomeID
		new_generation = new_generation.sort_values('chromosomeID')
		new_generation.index = np.arange(population_size)
		
		assert((new_generation['generationID'].mean() == generationID)).all(),\
			"The GenerationID did not update: should be {}; but is {}".format(
				generationID, new_generation['generationID'].mean())
		
		generation = train_generation(new_generation, clargs, machines)
		
		info_message('Time for Generation{}: {} minutes'.format(generationID, 
											(time() - start_while)//60))
		
		fitnesses = [chrom.fitness for _, chrom in generation.iterrows()]
		
		new_best_fitness = max(fitnesses)
		
		if clargs.verbose:
			info_message('For Generation: {}, the best fitness was {}'.format(
					generationID, new_best_fitness))
		
		best_fitness.append(new_best_fitness)
		
		if make_plots:
			plt.plot(best_fitness, color="c")
			plt.xlim([0, num_generations])
			fig.canvas.draw()