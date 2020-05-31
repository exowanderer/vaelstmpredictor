import sys, os.path
sys.path.append(os.path.abspath('../'))

from database import db, Chromosome
import numpy as np
import pandas as pd
import random
import time
from flask import json
import os

def debug_message(message, end = '\n'):
    print('[DEBUG] {}'.format(message), end = end)

def warning_message(message, end = '\n'):
    print('[WARNING] {}'.format(message), end = end)

def info_message(message, end = '\n'):
    print('[INFO] {}'.format(message), end = end)

def create_blank_dataframe(clargs):

    generation = pd.DataFrame()
    population_size = clargs.population_size

    generation['date_created'] = np.ones(population_size, dtype=int)*int(time.time())
    generation['date_trained'] = np.zeros(population_size, dtype=int) -1
    generation['date_taken'] = np.zeros(population_size, dtype=int) -1
    generation['fitness'] = np.zeros(population_size, dtype = float) -1.0
    generation['val_fitness'] = np.zeros(population_size, dtype = float) -1.0
    generation['train_reconstruction_loss'] = np.zeros(population_size, dtype = float) -1.0
    generation['train_kl_loss'] = np.zeros(population_size, dtype = float) -1.0
    generation['test_reconstruction_loss'] = np.zeros(population_size, dtype = float) -1.0
    generation['test_kl_loss'] = np.zeros(population_size, dtype = float) -1.0
    generation['info'] = np.array(['']*population_size)

    generation['cross_prob'] = np.ones(population_size, dtype=float) * clargs.cross_prob
    generation['mutate_prob'] = np.ones(population_size, dtype=float) * clargs.mutate_prob
    generation['population_size'] = np.ones(population_size, dtype=int) * clargs.population_size
    generation['batch_size'] = np.ones(population_size, dtype=int) * clargs.batch_size
    generation['num_epochs'] = np.ones(population_size, dtype=int) * clargs.num_epochs
    generation['chroms_per_loop'] = np.ones(population_size, dtype=int) * clargs.chroms_per_loop

    generation['num_cnn_encoder'] = np.ones(population_size, dtype=int)
    generation['size_kernel_encoder'] = [np.array([], dtype=int)]*population_size
    generation['size_pool_encoder'] = [np.array([], dtype=int)]*population_size
    generation['size_filter_encoder'] = [np.array([], dtype=int)]*population_size
    generation['batchnorm_encoder'] = [np.array([], dtype=int)]*population_size
    generation['num_dnn_encoder'] = np.ones(population_size, dtype=int)
    generation['size_dnn_encoder'] = [np.array([], dtype=int)]*population_size
    generation['l1_dnn_encoder'] = [np.array([], dtype=int)]*population_size

    # generation['num_cnn_decoder'] = np.ones(population_size, dtype=int)
    # generation['size_kernel_decoder'] = [np.array([], dtype=int)]*population_size
    # generation['size_pool_decoder'] = [np.array([], dtype=int)]*population_size
    # generation['size_filter_decoder'] = [np.array([], dtype=int)]*population_size
    # generation['num_dnn_decoder'] = np.ones(population_size, dtype=int)
    # generation['size_dnn_decoder'] = [np.array([], dtype=int)]*population_size

    generation['size_latent'] = np.ones(population_size, dtype=int)
    generation['size_resnet'] = np.zeros(population_size, dtype=int)

    return generation

def load_generation_from_sql(array_genes_sizes):
    generation = pd.read_sql(Chromosome.query.statement, db.session.bind)
    for param in list(array_genes_sizes.keys()):
        generation[param] = generation[param].apply(json.loads).apply(np.array)
    return generation

def add_generation_to_sql(generation, array_genes_sizes):
    if type(generation) == pd.Series:
        generation = generation.to_frame().T
    for param in list(array_genes_sizes.keys()):
        generation[param] = generation[param].apply(np.ndarray.tolist).apply(json.dumps)

    if "id" in generation:
        generation = generation.drop("id", axis=1)
    generation.to_sql('Chromosome', db.engine, if_exists='append', index=False)

def generate_random_chromosomes(clargs, array_genes_sizes):

    # create blank dataframe with full SQL database required entrie
    generation = create_blank_dataframe(clargs)
    population_size = clargs.population_size

    generation['num_cnn_encoder'] = loguniform(low=clargs.min_cnn_layers,
                                               high=clargs.max_cnn_layers,
                                               size = population_size)
    # generation['num_cnn_decoder'] = loguniform(low=clargs.min_cnn_layers,
    #                                           high=clargs.max_cnn_layers,
    #                                           size = population_size)
    generation['num_dnn_encoder'] = loguniform(low=clargs.min_dnn_layers,
                                               high=clargs.max_dnn_layers,
                                               size = population_size)
    # generation['num_dnn_decoder'] = loguniform(low=clargs.min_dnn_layers,
    #                                           high=clargs.max_dnn_layers,
    #                                           size = population_size)
    generation['size_latent'] = loguniform(low=clargs.min_latent_layers,
                                           high=clargs.max_latent_layers,
                                           size = population_size)
    generation['size_resnet'] = loguniform(low=clargs.min_resnet,
                                           high=clargs.max_resnet,
                                           size = population_size)

    for i in range(population_size):
        generation["size_kernel_encoder"].iat[i] = loguniform(low=clargs.min_kernel_size,
                                                                   high=clargs.max_kernel_size,
                                                                   size = (generation.loc[i, 'num_cnn_encoder']))
        generation["size_pool_encoder"].iat[i] = loguniform(low=clargs.min_pool_size,
                                                                   high=clargs.max_pool_size,
                                                                   size = (generation.loc[i, 'num_cnn_encoder']))
        generation["size_filter_encoder"].iat[i] = loguniform(low=clargs.min_filter_size,
                                                                   high=clargs.max_filter_size,
                                                                   size = (generation.loc[i, 'num_cnn_encoder']))
        generation["batchnorm_encoder"].iat[i] = loguniform(low=clargs.min_batchnorm,
                                                                   high=clargs.max_batchnorm,
                                                                   size = (generation.loc[i, 'num_cnn_encoder']))

        # generation["size_kernel_decoder"].iat[i] = loguniform(low=clargs.min_kernel_size,
        #                                                           high=clargs.max_kernel_size,
        #                                                           size = (generation.loc[i, 'num_cnn_decoder']))
        # generation["size_pool_decoder"].iat[i] = loguniform(low=clargs.min_pool_size,
        #                                                           high=clargs.max_pool_size,
        #                                                           size = (generation.loc[i, 'num_cnn_decoder']))
        # generation["size_filter_decoder"].iat[i] = loguniform(low=clargs.min_filter_size,
        #                                                           high=clargs.max_filter_size,
        #                                                           size = (generation.loc[i, 'num_cnn_decoder']))

        generation["size_dnn_encoder"].iat[i] = loguniform(low=clargs.min_dnn_size,
                                                                high=clargs.max_dnn_size,
                                                                size = (generation.loc[i, 'num_dnn_encoder']))
        generation["l1_dnn_encoder"].iat[i] = loguniform(low=clargs.min_l1,
                                                                high=clargs.max_l1,
                                                                size = (generation.loc[i, 'num_dnn_encoder']))

        # generation["size_dnn_decoder"].iat[i] = loguniform(low=clargs.min_dnn_size,
        #                                                         high=clargs.max_dnn_size,
        #                                                         size = (generation.loc[i, 'num_dnn_decoder']))

    add_generation_to_sql(generation, array_genes_sizes)

def loguniform(low=0, high=1, size=None, dtype=int):
    if dtype==int:
        high += 1
    return np.exp(np.random.uniform(np.log(low+1), np.log(high+1), size)).astype(dtype) -1

def train_generation(array_genes_sizes, sleep_time=30):
    while True:
        print("Waiting for Chromosomes to be Trained...")
        time.sleep(sleep_time)
        db.session.close()
        c = Chromosome.query.filter(Chromosome.date_trained <= 0).first()
        if c == None:
            print("All Chromosomes have been Trained")
            break

    generation = load_generation_from_sql(array_genes_sizes)
    return generation

def select_parents(generation):
    rank_fit = generation["fitness"].rank(ascending=True)
    rank_sum = rank_fit.sum()

    p1_num = np.random.uniform(0, rank_sum)
    p2_num = np.random.uniform(0, rank_sum)
    parent1 = None
    parent2 = None

    count = 0
    for index, rank in rank_fit.items():
        count += rank
        if(parent1 is None and count >= p1_num):
            parent1 = generation.iloc[index]
        if(parent2 is None and count >= p2_num):
            parent2 = generation.iloc[index]
        if(parent1 is not None and parent2 is not None):
            break

    return parent1, parent2

def cross_over(parent1, parent2, cross_prob, param_choices, array_genes_sizes):
    if random.random() <= cross_prob:
        child = parent1.to_dict()
        child["date_trained"] = -1
        child["date_taken"] = -1
        child["fitness"] = -1
        child["val_fitness"] = -1

        for param in param_choices:
            if(param in list(array_genes_sizes.keys())):
                continue
            gene_chosen = random.choice([parent1[param], parent2[param]])
            child[param] = gene_chosen

        for param in list(array_genes_sizes.keys()):
            size = child[array_genes_sizes[param]]
            new_array_gene = np.array([0]*size, dtype=int)
            for i in range(size):
                array_p1 = parent1[param]
                array_p2 = parent2[param]
                if(len(array_p1) <= i):
                    new_array_gene[i] = array_p2[i]
                elif(len(array_p2) <= i):
                    new_array_gene[i] = array_p1[i]
                else:
                    new_array_gene[i] = random.choice([array_p1[i], array_p2[i]])
            child[param] = new_array_gene

        child['info'] = 'Child of {} and {}'.format(parent1["id"], parent2["id"])
        return pd.Series(child)

    print(parent1)
    print(parent2)
    child = random.choice([parent1, parent2]).to_dict()
    child['info'] = 'Descendant of {}'.format(child["id"])
    child["date_created"] = int(time.time())
    del child["id"]

    return pd.Series(child)

def mutate(child, mutate_prob, param_choices, array_genes_sizes):
    mutation_happened = False
    child = child.to_dict()

    for param, (range_change, min_val, max_val) in param_choices.items():
        if(param not in list(array_genes_sizes.keys())):
            if(random.random() <= mutate_prob):
                mutation_happened = True
                # Compute delta_param step
                if(type(min_val) == int):
                    change_p = np.random.uniform(-range_change-0.5, range_change+0.5)
                    change_p = np.round(change_p)
                else:
                    change_p = np.random.uniform(-range_change, range_change)

                # Add delta_param to param
                current_p = child[param] + change_p

                # If param less than `min_val`, then set param to `min_val`
                current_p = np.max([current_p, min_val])
                current_p = np.min([current_p, max_val])

                if type(min_val) == int:
                    current_p = round(current_p)

                # All params must be integer sized: round and convert
                child[param] = current_p


    for param, (range_change, min_val, max_val) in param_choices.items():
        if(param in list(array_genes_sizes.keys())):
            size = child[array_genes_sizes[param]]

            #Fix size if it got mutated
            while(len(child[param]) != size):
                if len(child[param]) > size:
                    del_index = random.choice(range(len(child[param])))
                    child[param] = np.delete(child[param], del_index).astype(type(min_val))
                elif len(child[param]) < size:
                    if len(child[param]) != 0:
                        dup_index = random.choice(range(len(child[param])))
                        child[param] = np.append(child[param], child[param][dup_index]).astype(type(min_val))
                    else:
                        new_val = min_val + (random.random() * (max_val - min_val))
                        if type(min_val) == int:
                            new_val = round(new_val)

                        child[param] = np.append(child[param], new_val).astype(type(min_val))

            if(random.random() <= mutate_prob and len(child[param]) > 0):
                    mutation_happened = True

                    change_index = random.choice(range(len(child[param])))
                    if(type(min_val) == int):
                        change_p = np.random.uniform(-range_change-0.5, range_change+0.5)
                        change_p = np.round(change_p)
                    else:
                        change_p = np.random.uniform(-range_change, range_change)

                    current_p = child[param][change_index] + change_p
                    current_p = np.max([current_p, int(min_val)])
                    if type(min_val) == int:
                        current_p = round(current_p)

                    child[param][change_index] = current_p

    if(mutation_happened):
        child['info'] = child['info']+" [Mutated]"
        child["date_created"] = int(time.time())
        child["date_trained"] = -1
        child["date_taken"] = -1
        child["fitness"] = -1
        child["val_fitness"] = -1
        if "id" in child: del child["id"]

    return pd.Series(child)
