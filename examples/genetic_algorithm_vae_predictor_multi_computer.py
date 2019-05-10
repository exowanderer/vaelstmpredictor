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

def debug_message(message): print('[DEBUG] {}'.format(message))
def info_message(message): 
	# print('[INFO] {}'.format(message))
	pass

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
	parser.add_argument('--num_epochs', type=int, default=200,
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
	parser.add_argument('--population_size', type=int, default=200,
				help='size of the population to evolve; '\
						'preferably divisible by 2')
	parser.add_argument('--num_generations', type=int, default=100,
				help='number of generations for genetic algorithm')
	parser.add_argument('--verbose', action='store_true',
				help='print more [INFO] and [DEBUG] statements')
	parser.add_argument('--make_plots', action='store_true',
				help='make plots of growth in the best_loss over generations')
	parser.add_argument('--port', type=int, default=22,
				help='IP port over which to ssh')
	debug_message(1)
	clargs = parser.parse_args()
	debug_message(2)
	for key,val in clargs.__dict__.items(): 
		if 'dir' in key: 
			if not os.path.exists(val): 
				os.mkdir(val)
	debug_message(3)
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	hostname = s.getsockname()[0]
	s.close()
	debug_message(4)
	clargs.hostname = hostname
	debug_message(5)
	run_name = clargs.run_name
	num_epochs = clargs.num_epochs
	cross_prob = clargs.cross_prob
	mutate_prob = clargs.mutate_prob
	population_size = clargs.population_size
	num_generations = clargs.num_generations
	verbose = clargs.verbose
	make_plots = clargs.make_plots
	debug_message(6)
	clargs.data_type = 'MNIST'
	data_instance = MNISTData(batch_size = clargs.batch_size)
	debug_message(7)
	n_train, n_features = data_instance.data_train.shape
	n_test, n_features = data_instance.data_valid.shape
	debug_message(8)
	clargs.original_dim = n_features
	debug_message(9)
	clargs.time_stamp = int(time())
	clargs.run_name = '{}_{}_{}'.format(clargs.run_name, 
								clargs.data_type, clargs.time_stamp)
	debug_message(10)
	if verbose: print('\n\n[INFO] Run Base Name: {}\n'.format(clargs.run_name))
	debug_message(11)
	clargs.n_labels = len(np.unique(data_instance.train_labels))
	debug_message(12)
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
	debug_message(13)
	# generation = convert_dtypes(generation)
	debug_message(14)
	generationID = 0
	generation = train_generation(generation, clargs)
	debug_message(15)
	# generation = convert_dtypes(generation)
	debug_message(16)
	best_fitness = []
	fitnesses = [chromosome.fitness for _, chromosome in generation.iterrows()]
	debug_message(17)
	new_best_fitness = max(fitnesses)
	debug_message(18)
	if clargs.verbose:
		info_message('For Generation: {}, the best fitness was {}'.format(
				generationID, new_best_fitness))
	debug_message(19)
	best_fitness.append(new_best_fitness)
	debug_message(20)
	if make_plots:
		fig = plt.gcf()
		fig.show()
	debug_message(21)
	param_choices = {'num_vae_layers': (1,1), 
					 'num_dnn_layers': (1,1), 
					 'size_vae_latent': (10,1), 
					 'size_vae_hidden': (50,1), 
					 'size_dnn_hidden': (50,1)}
	debug_message(22)
	start = time()
	# while gen_num < num_generations:
	for generationID in range(1,num_generations):
		start_while = time()
		# Create new generation
		debug_message(generationID,23)
		new_generation = create_blank_dataframe(generationID, population_size)
		# new_generation = convert_dtypes(new_generation)
		debug_message(generationID,24)
		for chromosomeID in tqdm(range(population_size)):
			parent1, parent2 = select_parents(generation)
			debug_message(generationID,chromosomeID,25)
			child, crossover_happened = cross_over(parent1, parent2, 
											cross_prob, param_choices.keys(), 
											verbose=verbose)
			debug_message(generationID,chromosomeID,26)
			child.generationID = int(generationID)
			child.chromosomeID = int(chromosomeID)
			child.fitness = -1.0
			debug_message(generationID,chromosomeID,27)
			child, mutation_happened = mutate(child, mutate_prob, 
											param_choices, verbose=verbose)
			debug_message(generationID,chromosomeID,28)
			new_generation.set_value(child.Index, 'isTrained', 2)
			child.isTrained = mutation_happened*crossover_happened
			debug_message(generationID,chromosomeID,29)
			info_message('Adding Chromosome:\n{}'.format(child))
			new_generation.iloc[chromosomeID] = child
		debug_message(generationID,28)
		# Re-sort by chromosomeID
		new_generation = new_generation.sort_values('chromosomeID')
		new_generation.index = np.arange(population_size)
		debug_message(generationID,29)
		assert((new_generation['generationID'].values == generationID)).all(),\
			"The GenerationID did not update: should be {}; but is {}".format(
				generationID, generation['generationID'].values)
		debug_message(generationID,30)
		generation = train_generation(new_generation, clargs)
		debug_message(generationID,31)
		info_message('Time for Generation{}: {} minutes'.format(generationID, 
											(time() - start_while)//60))
		debug_message(generationID,32)
		fitnesses = [chrom.fitness for _, chrom in generation.iterrows()]
		debug_message(generationID,33)
		new_best_fitness = max(fitnesses)
		debug_message(generationID,34)
		if clargs.verbose:
			info_message('For Generation: {}, the best fitness was {}'.format(
					generationID, new_best_fitness))
		debug_message(generationID,35)
		best_fitness.append(new_best_fitness)
		debug_message(generationID,36)
		if make_plots:
			plt.plot(best_fitness, color="c")
			plt.xlim([0, num_generations])
			fig.canvas.draw()
		debug_message(generationID,37)