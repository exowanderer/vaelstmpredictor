# from https://github.com/philippesaade11/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --iterations 500 --population_size 10 --num_epochs 200
import argparse
import socket
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time, sleep
import json

from GeneticAlgorithm import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--generationID', type=int, required=True,
		help='Chromosome Generation ID')
	parser.add_argument('--chromosomeID', type=int, required=True,
		help='Chromosome Chromosome ID')
	parser.add_argument('--num_vae_layers', type=int, required=True,
		help='Depth of the VAE')
	parser.add_argument('--num_dnn_layers', type=int, required=True,
		help='Depth of the DNN')
	parser.add_argument('--size_vae_latent', type=int, required=True,
		help='Size of the VAE Latent Layer')
	parser.add_argument('--size_vae_hidden', type=int, required=True,
		help='Size of the VAE Hidden Layer')
	parser.add_argument('--size_dnn_hidden', type=int, required=True,
		help='Size of the DNN Hidden Layer')

	params = parser.parse_args()
	var_params = var(params)
	
	num_class = 10 # MNIST

	original_dim = 784 # MNIST
	vae_hidden_dims = [params.size_vae_hidden]*params.num_vae_layers
	dnn_hidden_dims = [params.size_dnn_hidden]*params.num_dnn_layers
    vae_latent_dim = params.size_vae_latent
    dnn_out_dim = num_class
    dnn_latent_dim = num_class - 1
    batch_size = 128
    dnn_log_var_prior = 0.0
    optimizer = 'adam'
    use_prev_input = False
    predictor_type = 'classification'

    clargs = None
    data_instance = None
    generationID = 0
    chromosomeID = 0
    vae_kl_weight = 1.0
    dnn_weight = 1.0
    dnn_kl_weight = 1.0
    verbose = True
    
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	hostname = s.getsockname()[0]
	s.close()
	
	print('\n\nParams for this VAE_NN:')
	for key,val in var_params.items():
		print('{:20}{}'.format(key,val))

	chrom_params['verbose'] = True

	chrom = Chromosome(**chrom_params)
	chrom.verbose = True
	chrom.train(verbose=True)
	print("done")
