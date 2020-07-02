import sys
import os.path
sys.path.append(os.path.abspath('../'))

import numpy as np
import socket
from time import time
from tqdm import tqdm
import argparse
import pandas as pd
from database import Variables, Chromosome, db

from GeneticAlgorithm_VAE import (generate_random_chromosomes, train_generation,
                                  create_blank_dataframe, select_parents, cross_over,
                                  mutate, load_generation_from_sql, add_generation_to_sql)


def debuge_message(message): print('[DEBUG] {}'.format(message))

def info_message(message): print('[INFO] {}'.format(message))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_prob', type=float, default=0.7,
                        help='Probability of crossover between generations')
    parser.add_argument('--mutate_prob', type=float, default=0.01,
                        help='Probability of mutation for each member')
    parser.add_argument('--population_size', type=int, default=100,
                        help='size of the population to evolve; preferably divisible by 2')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--chroms_per_loop', type=int, default=10,
                        help='Number of Chromosomes to create at a time per loop')
    parser.add_argument('--sleep_time', type=int, default=30,
                        help='Interval time between training checks')

    parser.add_argument('--max_dnn_layers', type=int, default=2,
                        help='Maximum number of VAE hidden layers')
    parser.add_argument('--min_dnn_layers', type=int, default=0,
                        help='Minimum number of DNN hidden layers')

    parser.add_argument('--max_cnn_layers', type=int, default=4,
                        help='Maximum number of Convolution layers')
    parser.add_argument('--min_cnn_layers', type=int, default=1,
                        help='Minimum number of Convolution layers')

    parser.add_argument('--max_latent_layers', type=int, default=256,
                        help='Maximum number of nodes in the latent vector')
    parser.add_argument('--min_latent_layers', type=int, default=1,
                        help='Minimum number of nodes in the latent vector')

    parser.add_argument('--max_resnet', type=int, default=3,
                        help='Maximum number of Resnets')
    parser.add_argument('--min_resnet', type=int, default=0,
                        help='Minimum number of Resnets')

    parser.add_argument('--max_kernel_size', type=int, default=6,
                        help='Maximum kernel size (x2 +1)')
    parser.add_argument('--min_kernel_size', type=int, default=0,
                        help='Minimum kernel size (x2 +1)')

    parser.add_argument('--max_pool_size', type=int, default=4,
                        help='Maximum Max Pooling size (x2)')
    parser.add_argument('--min_pool_size', type=int, default=0,
                        help='Minimum Max Pooling size (x2)')

    parser.add_argument('--max_filter_size', type=int, default=256,
                        help='Maximum number of filters for each convolution layer')
    parser.add_argument('--min_filter_size', type=int, default=1,
                        help='Minimum number of filters for each convolution layer')

    parser.add_argument('--max_batchnorm', type=int, default=1,
                        help='Maximum number of batchnorm for each convoltution layer')
    parser.add_argument('--min_batchnorm', type=int, default=0,
                        help='Minimum number of batchnorm for each convoltution layer')

    parser.add_argument('--max_dnn_size', type=int, default=256,
                        help='Maximum number of nodes for each dense layer')
    parser.add_argument('--min_dnn_size', type=int, default=1,
                        help='Minimum number of nodes for each dense layer')

    parser.add_argument('--max_l1', type=int, default=3,
                        help='Maximum index of [0, 0.01, 0.1, 1, ...]')
    parser.add_argument('--min_l1', type=int, default=0,
                        help='Minimum index of [0, 0.01, 0.1, 1, ...]')

    clargs = parser.parse_args()

    param_choices = {'num_cnn_encoder': (1, 1, 5),
                     'size_kernel_encoder': (1, 0, 6),
                     'size_pool_encoder': (1, 0, 4),
                     'size_filter_encoder': (10, 1, 500),
                     'num_dnn_encoder': (1, 1, 5),
                     'size_dnn_encoder': (10, 1, 500),

                    #  'num_cnn_decoder': (1, 1),
                    #  'size_kernel_decoder': (1, 0),
                    #  'size_pool_decoder': (1, 0),
                    #  'size_filter_decoder': (10, 1),
                    #  'num_dnn_decoder': (1, 1),
                    #  'size_dnn_decoder': (10, 1),

                     'size_latent': (10, 1, 500),

                     'batchnorm_encoder': (1, 0, 1),
                     'l1_dnn_encoder': (1, 0, 3),
                     'size_resnet': (1, 0, 3),}

    array_genes_sizes = {"size_kernel_encoder": 'num_cnn_encoder',
                   "size_pool_encoder": 'num_cnn_encoder',
                   "size_filter_encoder": 'num_cnn_encoder',
                   "batchnorm_encoder": 'num_cnn_encoder',

                #   "size_kernel_decoder": 'num_cnn_decoder',
                #   "size_pool_decoder": 'num_cnn_decoder',
                #   "size_filter_decoder": 'num_cnn_decoder',

                #   "size_dnn_decoder": 'num_dnn_decoder',
                   "size_dnn_encoder": 'num_dnn_encoder',
                   "l1_dnn_encoder": 'num_dnn_encoder',}

    generation = Chromosome.query.all()
    if(len(generation) < clargs.population_size):
        generate_random_chromosomes(clargs, array_genes_sizes)
    else:
        print("Loading Generation From DB")

    generation = train_generation(array_genes_sizes, sleep_time=clargs.sleep_time)

    while True:
        new_gen = pd.DataFrame()
        for _ in tqdm(range(clargs.chroms_per_loop)):
            parent1, parent2 = select_parents(generation)
            child = cross_over(parent1, parent2,
                               clargs.cross_prob,
                               list(param_choices.keys()), array_genes_sizes)

            child = mutate(child,
                           clargs.mutate_prob,
                           param_choices, array_genes_sizes)

            new_gen = new_gen.append(child, ignore_index=True)
        add_generation_to_sql(new_gen, array_genes_sizes)

        print("Added {} Chromosomes".format(clargs.chroms_per_loop))
        generation = train_generation(array_genes_sizes, sleep_time=clargs.sleep_time)
