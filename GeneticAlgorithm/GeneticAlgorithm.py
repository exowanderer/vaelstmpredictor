# from https://github.com/philippesaade11/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --iterations 500 --population_size 10 --num_epochs 200
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
# import random

from contextlib import redirect_stdout

from keras import backend as K
from keras.utils import to_categorical

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros, random
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

def generate_random_chromosomes(population_size, clargs, data_instance, 
                        start_small = False, init_large = False,
                        vae_kl_weight = 1.0, input_size = None, 
                        max_vae_hidden_layers = 5, max_dnn_hidden_layers = 5, 
                        dnn_weight = 1.0, dnn_kl_weight = 1.0, 
                        min_vae_hidden1 = 2, min_vae_latent = 2, 
                        min_dnn_hidden1 = 2, max_vae_hidden = 1024, 
                        max_vae_latent = 1024, max_dnn_hidden = 1024, 
                        verbose=False, TrainFunction = None):
    
    start_small = start_small or clargs.start_small 
    init_large = init_large or clargs.init_large
    vae_kl_weight = vae_kl_weight or clargs.vae_kl_weight 
    input_size  = input_size or clargs.original_dim 
    max_vae_hidden_layers = max_vae_hidden_layers \
                                or clargs.max_vae_hidden_layers 
    max_dnn_hidden_layers = max_dnn_hidden_layers \
                                or clargs.max_dnn_hidden_layers 
    dnn_weight = dnn_weight or clargs.dnn_weight 
    dnn_kl_weight = dnn_kl_weight or clargs.dnn_kl_weight 
    min_vae_hidden1 = min_vae_hidden1 or clargs.min_vae_hidden1 
    min_vae_latent = min_vae_latent or clargs.min_vae_latent 
    min_dnn_hidden1 = min_dnn_hidden1 or clargs.min_dnn_hidden1 
    max_vae_hidden = max_vae_hidden or clargs.max_vae_hidden 
    max_vae_latent = max_vae_latent or clargs.max_vae_latent 
    max_dnn_hidden = max_dnn_hidden or clargs.max_dnn_hidden 
    verbose = verbose or clargs.verbose

    generationID = 0
    generation_0 = []
    for chromosomeID in range(population_size):
        # explicit defaults
        vae_hidden_dims = configure_multi_hidden_layers(max_vae_hidden_layers, 
                                input_size, min_vae_hidden1, max_vae_hidden,
                                start_small = start_small, 
                                init_large = init_large)
        
        vae_latent_dim = random.randint(min_vae_latent, max_vae_latent)
        
        dnn_hidden_dims = configure_multi_hidden_layers(max_dnn_hidden_layers, 
                                input_size, min_dnn_hidden1, max_dnn_hidden,
                                start_small = start_small, 
                                init_large = init_large)

        params_dict =  {'clargs':clargs, 
                        'verbose':verbose, 
                        'data_instance':data_instance, 
                        'generationID':generationID, 
                        'chromosomeID':chromosomeID,
                        'vae_kl_weight':vae_kl_weight, 
                        'dnn_weight':dnn_weight,
                        'dnn_kl_weight':dnn_kl_weight
                       }
        
        params_dict['vae_latent_dim'] = vae_latent_dim
        params_dict['vae_hidden_dims'] = vae_hidden_dims
        params_dict['dnn_hidden_dims'] = dnn_hidden_dims

        # for k, layer_size in enumerate(vae_hidden_dims):
        #     params_dict['size_vae_hidden{}'.format(k)] = layer_size
        # for k, layer_size in enumerate(dnn_hidden_dims):
        #     params_dict['size_dnn_hidden{}'.format(k)] = layer_size

        chrom = Chromosome(**params_dict)
        generation_0.append(chrom)

    if(TrainFunction is None):
        for chrom in generation_0:
            chrom.train()
    else :
        TrainFunction(generation_0, clargs)
        
    return generation_0

def select_parents(generation):
    total_fitness = sum(chrom.fitness for chrom in generation)
    #Generate two random numbers between 0 and total_fitness 
    #   not including total_fitness
    rand_parent1 = random.random()*total_fitness
    rand_parent2 = random.random()*total_fitness
    parent1 = None
    parent2 = None
    
    fitness_count = 0
    for chromosome in generation:
        fitness_count += chromosome.fitness
        if(parent1 == None and fitness_count >= rand_parent1):
            parent1 = chromosome
        if(parent2 == None and fitness_count >= rand_parent2):
            parent2 = chromosome
        if(parent1 != None and parent2 != None):
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

def cross_over(parent1, parent2, prob, verbose=False):
    if verbose:
        print('Crossing over with Parent {} and Parent {}'.format(
                        parent1.chromosomeID, parent2.chromosomeID))

    static_params_ = {  'clargs':parent1.clargs, 
                        'data_instance':parent1.data_instance, 
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
        
        child1 = Chromosome(**params1)
        child2 = Chromosome(**params2)

        return child1, child2, crossover_happened
    
    return parent1, parent2, not crossover_happened

def mutate(child, prob, range_change = 25, forced_evolve = False, 
            min_layer_size = 2, verbose = False):
    
    # explicit declaration
    zero = 0 

    if verbose:
        print('Mutating Child {} in Generation {}'.format(child.chromosomeID, 
                                                         child.generationID))
    
    mutation_happened = False
    for param in child.params_dict.keys():
        if(random.random() <= prob):
            mutation_happened = True

            # Compute delta_param step
            change_p = np.random.uniform(-range_change, range_change)

            if forced_evolve and child.params_dict[param] == zero:
                # if the layer is empty, then force a mutation
                change_p = min([min_layer_size, abs(change_p)])

            # Add delta_param to param
            current_p = child.params_dict[param] + change_p
            
            # If layer size param is negative, then set layer size to zero
            child.params_dict[param] = np.max([child.params_dict[param],zero])

            # All layers must be integer sized: round and convert
            child.params_dict[param] = np.int(np.round(current_p))

    return child, mutation_happened

def save_generation_to_tree(generation, verbose = False):
    generation_dict = {}
    if verbose: print('[INFO] Current Generation: ' )

    for ID, member in enumerate(generation):
        if ID not in generation_dict.keys(): generation_dict[ID] = {}
        
        if verbose: 
            print('memberID: {}'.format(ID))
            print('Fitness: {}'.format(member.fitness))
            for key,val in member.params_dict.items():
                print('\t{}: {}'.format(key, val))

        generation_dict[ID]['params'] = member.params_dict
        generation_dict[ID]['fitness'] = member.fitness
    
    return generation_dict
