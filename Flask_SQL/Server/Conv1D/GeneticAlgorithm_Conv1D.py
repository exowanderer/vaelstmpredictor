import sys, os.path
sys.path.append(os.path.abspath('../'))

from database import db, Chromosome
import numpy as np
import pandas as pd
from numpy import random
from time import sleep
from flask import json
import os

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

def create_blank_dataframe(generationID, population_size):

    generation = pd.DataFrame()

    zeros = np.zeros(population_size, dtype = int)
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
    generation['dnn_log_var_prior'] = zeros
    generation['hostname'] = ['127.0.0.1']*population_size
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
    generation['run_time'] = np.float32(zeros)
    generation['start_time'] = np.float32(zeros)
    generation['end_time'] = np.float32(zeros)
    generation['train_file'] = ['train_file']*population_size
    generation['dnn_weight'] = np.float32(zeros)
    generation['dnn_kl_weight'] = np.float32(zeros)
    generation['vae_kl_weight'] = np.float32(zeros)
    generation['vae_weight'] = np.float32(zeros)
    generation['w_kl_anneal'] = zeros
    generation['num_conv_layers'] = zeros
    generation['size_kernel'] = [np.array([], dtype=int)]*population_size
    generation['size_pool'] = [np.array([], dtype=int)]*population_size
    generation['size_filter'] = [np.array([], dtype=int)]*population_size
    generation['info'] = ['']*population_size
    generation['val_vae_reconstruction_loss'] = np.float32(zeros)
    generation['val_vae_latent_args_loss'] = np.float32(zeros)
    generation['val_dnn_latent_layer_loss'] = np.float32(zeros)
    generation['val_dnn_predictor_layer_loss'] = np.float32(zeros)

    return generation

def generate_random_chromosomes(population_size, clargs, geneationID = 0,
                        min_vae_hidden_layers = 1, max_vae_hidden_layers = 5,
                        min_dnn_hidden_layers = 1, max_dnn_hidden_layers = 5,
                        min_vae_hidden = 2, max_vae_hidden = 1024,
                        min_dnn_hidden = 2, max_dnn_hidden = 1024,
                        min_vae_latent = 2, max_vae_latent = 1024,
                        min_conv_layers = 1, max_conv_layers = 5,
                        min_kernel_size = 0, max_kernel_size = 6,
                        min_pool_size = 0, max_pool_size = 4,
                        min_filter_size = 1, max_filter_size = 128,
                        verbose=False):

    # create blank dataframe with full SQL database required entrie
    generation = create_blank_dataframe(geneationID, population_size)

    # Add weights from clargs
    generation['dnn_kl_weight'] = np.repeat(clargs.dnn_kl_weight, population_size)
    generation['dnn_weight'] = np.repeat(clargs.dnn_weight, population_size)
    generation['vae_kl_weight'] = np.repeat(clargs.vae_kl_weight, population_size)
    generation['vae_weight'] = np.repeat(clargs.vae_weight, population_size)

    # Overwrite chromosome parameters to evolve with random choices
    vae_nLayers_choices = range(min_vae_hidden_layers, max_vae_hidden_layers)
    dnn_nLayers_choices = range(min_dnn_hidden_layers, max_dnn_hidden_layers)
    vae_latent_choices = range(min_vae_latent, max_vae_latent)
    vae_nUnits_choices = range(min_vae_hidden, max_vae_hidden)
    dnn_nUnits_choices = range(min_dnn_hidden, max_dnn_hidden)
    conv_nUnits_choices = range(min_conv_layers, max_conv_layers)
    kernel_nUnits_choices = range(min_kernel_size, max_kernel_size)
    pool_nUnits_choices = range(min_pool_size, max_pool_size)
    filter_nUnits_choices = range(min_filter_size, max_filter_size)

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
    generation['num_conv_layers'] = np.random.choice(conv_nUnits_choices,
                                                        size = population_size)

    for i in range(population_size):
        generation.set_value(i, "size_kernel",  np.random.choice(kernel_nUnits_choices, size = (generation.loc[i, 'num_conv_layers'])))
        generation.set_value(i, "size_pool",  np.random.choice(pool_nUnits_choices, size = (generation.loc[i, 'num_conv_layers'])))
        generation.set_value(i, "size_filter",  np.random.choice(filter_nUnits_choices, size = (generation.loc[i, 'num_conv_layers'])))

    return generation

def train_generation(generation, clargs, verbose=False, sleep_time=30, save_DB=True):
    generationID = generation['generationID'][0]
    print("Generation has "+str(len(generation))+" Chromosome")

    if(save_DB):
        for chromosome in generation.itertuples():
            print("Chromosome ID: "+str(chromosome.chromosomeID))
            chromosomeID = chromosome.chromosomeID
            generationID = chromosome.generationID #Get Global Generation ID
            fitness = chromosome.fitness
            run_name = clargs.run_name
            predictor_type = clargs.predictor_type
            batch_size = clargs.batch_size
            optimizer = clargs.optimizer
            num_epochs = clargs.num_epochs
            dnn_weight = chromosome.dnn_weight
            vae_weight = chromosome.vae_weight
            vae_kl_weight = chromosome.vae_kl_weight
            dnn_kl_weight = chromosome.dnn_kl_weight
            prediction_log_var_prior = clargs.prediction_log_var_prior
            patience = clargs.patience
            kl_anneal = clargs.kl_anneal
            w_kl_anneal = clargs.w_kl_anneal
            dnn_log_var_prior = clargs.dnn_log_var_prior
            log_dir = clargs.log_dir
            model_dir = clargs.model_dir
            table_dir = clargs.table_dir
            train_file = clargs.train_file
            cross_prob = clargs.cross_prob
            mutate_prob = clargs.mutate_prob
            population_size = clargs.population_size
            num_generations = clargs.num_generations
            start_time = chromosome.start_time
            end_time = chromosome.end_time
            run_time = chromosome.end_time - chromosome.start_time
            hostname = clargs.hostname
            val_vae_reconstruction_loss = chromosome.val_vae_reconstruction_loss
            val_vae_latent_args_loss = chromosome.val_vae_latent_args_loss
            val_dnn_latent_layer_loss = chromosome.val_dnn_latent_layer_loss
            val_dnn_predictor_layer_loss = chromosome.val_dnn_predictor_layer_loss
            num_vae_layers = chromosome.num_vae_layers
            num_dnn_layers = chromosome.num_dnn_layers
            size_vae_latent = chromosome.size_vae_latent
            size_vae_hidden = chromosome.size_vae_hidden
            size_dnn_hidden = chromosome.size_dnn_hidden
            num_conv_layers = chromosome.num_conv_layers
            size_kernel = json.dumps(chromosome.size_kernel.tolist())
            size_pool = json.dumps(chromosome.size_pool.tolist())
            size_filter = json.dumps(chromosome.size_filter.tolist())
            info = chromosome.info

            c = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chromosomeID, Chromosome.generationID == generationID).first()
            if(c == None):
                chrom = Chromosome(chromosomeID = chromosomeID,
                                    generationID = generationID,
                                    fitness = fitness,
                                    run_name = run_name,
                                    predictor_type = predictor_type,
                                    batch_size = batch_size,
                                    optimizer = optimizer,
                                    num_epochs = num_epochs,
                                    dnn_weight = dnn_weight,
                                    vae_weight = vae_weight,
                                    vae_kl_weight = vae_kl_weight,
                                    dnn_kl_weight = dnn_kl_weight,
                                    prediction_log_var_prior = prediction_log_var_prior,
                                    patience = patience,
                                    kl_anneal = kl_anneal,
                                    w_kl_anneal = w_kl_anneal,
                                    dnn_log_var_prior = dnn_log_var_prior,
                                    log_dir = log_dir,
                                    model_dir = model_dir,
                                    table_dir = table_dir,
                                    train_file = train_file,
                                    cross_prob = cross_prob,
                                    mutate_prob = mutate_prob,
                                    population_size = population_size,
                                    num_generations = num_generations,
                                    start_time = start_time,
                                    end_time = end_time,
                                    run_time = run_time,
                                    hostname = hostname,
                                    val_vae_reconstruction_loss = val_vae_reconstruction_loss,
                                    val_vae_latent_args_loss = val_vae_latent_args_loss,
                                    val_dnn_latent_layer_loss = val_dnn_latent_layer_loss,
                                    val_dnn_predictor_layer_loss = val_dnn_predictor_layer_loss,
                                    num_vae_layers = num_vae_layers,
                                    num_dnn_layers = num_dnn_layers,
                                    size_vae_latent = size_vae_latent,
                                    size_vae_hidden = size_vae_hidden,
                                    size_dnn_hidden = size_dnn_hidden,
                                    num_conv_layers = num_conv_layers,
                                    size_kernel = size_kernel,
                                    size_pool = size_pool,
                                    size_filter = size_filter,
                                    info = info)
                db.session.add(chrom)
            else :
                c.fitness = fitness
                c.run_name = run_name
                c.predictor_type = predictor_type
                c.batch_size = batch_size
                c.optimizer = optimizer
                c.num_epochs = num_epochs
                c.dnn_weight = dnn_weight
                c.vae_weight = vae_weight
                c.vae_kl_weight = vae_kl_weight
                c.dnn_kl_weight = dnn_kl_weight
                c.prediction_log_var_prior = prediction_log_var_prior
                c.patience = patience
                c.kl_anneal = kl_anneal
                c.w_kl_anneal = w_kl_anneal
                c.dnn_log_var_prior = dnn_log_var_prior
                c.log_dir = log_dir
                c.model_dir = model_dir
                c.table_dir = table_dir
                c.train_file = train_file
                c.cross_prob = cross_prob
                c.mutate_prob = mutate_prob
                c.population_size = population_size
                c.num_generations = num_generations
                c.start_time = start_time
                c.end_time = end_time
                c.run_time = run_time
                c.hostname = hostname
                c.val_vae_reconstruction_loss = val_vae_reconstruction_loss
                c.val_vae_latent_args_loss = val_vae_latent_args_loss
                c.val_dnn_latent_layer_loss = val_dnn_latent_layer_loss
                c.val_dnn_predictor_layer_loss = val_dnn_predictor_layer_loss
                c.num_vae_layers = num_vae_layers
                c.num_dnn_layers = num_dnn_layers
                c.size_vae_latent = size_vae_latent
                c.size_vae_hidden = size_vae_hidden
                c.size_dnn_hidden = size_dnn_hidden
                c.num_conv_layers = num_conv_layers
                c.size_kernel = size_kernel
                c.size_pool = size_pool
                c.size_filter = size_filter
                c.info = info
            db.session.commit()

    while True:
        print("Waiting for Chromosomes to be Trained in Generation "+str(generationID))
        sleep(sleep_time)
        c = db.session.query(Chromosome).filter(Chromosome.generationID == generationID, Chromosome.isTrained != 2).first()
        db.session.commit()
        print(c)
        if c == None:
            print("All Chromosomes for Generation {} have been Trained".format(generationID))
            break
    print("Create Generation "+str(generationID +1))

    for chrom in generation.itertuples():
        sql_chrom = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chrom.chromosomeID, Chromosome.generationID == generationID).first()
        assert(sql_chrom.isTrained == 2), "Finished training yet there's a chromosome with isTrained != 2"
        generation.set_value(chrom.chromosomeID, "isTrained", sql_chrom.isTrained)
        generation.set_value(chrom.chromosomeID, "fitness", sql_chrom.fitness)
        generation.set_value(chrom.chromosomeID, "val_dnn_latent_layer_loss", sql_chrom.val_dnn_latent_layer_loss)
        generation.set_value(chrom.chromosomeID, "val_dnn_predictor_layer_loss", sql_chrom.val_dnn_predictor_layer_loss)
        generation.set_value(chrom.chromosomeID, "val_vae_latent_args_loss", sql_chrom.val_vae_latent_args_loss)
        generation.set_value(chrom.chromosomeID, "val_vae_reconstruction_loss", sql_chrom.val_vae_reconstruction_loss)
    return generation

def refactor_weights(new_generation, generation):
    dnn_weights = []
    dnn_kl_weights = []
    vae_weights = []
    vae_kl_weights = []
    for chrom in generation.itertuples():
        val_loss = chrom.val_vae_reconstruction_loss + chrom.val_vae_latent_args_loss + chrom.val_dnn_latent_layer_loss + chrom.val_dnn_predictor_layer_loss
        dnn_weight_ = val_loss / chrom.val_dnn_predictor_layer_loss
        dnn_kl_weight_ = val_loss / chrom.val_dnn_latent_layer_loss
        vae_weight_ = val_loss / chrom.val_vae_reconstruction_loss
        vae_kl_weight_ = val_loss / chrom.val_vae_latent_args_loss
        dnn_weights.append(dnn_weight_)
        dnn_kl_weights.append(dnn_kl_weight_)
        vae_weights.append(vae_weight_)
        vae_kl_weights.append(vae_kl_weight_)
    dnn_weight_new = np.median(dnn_weights)
    dnn_kl_weight_new = np.median(dnn_kl_weights)
    vae_weight_new = np.median(vae_weights)
    vae_kl_weight_new = np.median(vae_kl_weights)
    print("Weights", dnn_weight_new, dnn_kl_weight_new, vae_weight_new, vae_kl_weight_new)
    for chrom in new_generation.itertuples():
        new_generation.set_value(chrom.chromosomeID, "dnn_weight", dnn_weight_new)
        new_generation.set_value(chrom.chromosomeID, "dnn_kl_weight", dnn_kl_weight_new)
        new_generation.set_value(chrom.chromosomeID, "vae_weight", vae_weight_new)
        new_generation.set_value(chrom.chromosomeID, "vae_kl_weight", vae_kl_weight_new)

def load_generation_from_sql(generationID, population_size):
    generation = create_blank_dataframe(generationID, population_size)
    for chrom in generation.itertuples():
        sql_chrom = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chrom.chromosomeID, Chromosome.generationID == generationID).first()
        generation.set_value(chrom.chromosomeID, "info", sql_chrom.info)
        generation.set_value(chrom.chromosomeID, "size_pool", np.array(json.loads(sql_chrom.size_filter)))
        generation.set_value(chrom.chromosomeID, "size_filter", np.array(json.loads(sql_chrom.size_filter)))
        generation.set_value(chrom.chromosomeID, "size_kernel", np.array(json.loads(sql_chrom.size_kernel)))
        generation.set_value(chrom.chromosomeID, "num_conv_layers", sql_chrom.num_conv_layers)
        generation.set_value(chrom.chromosomeID, "size_dnn_hidden", sql_chrom.size_dnn_hidden)
        generation.set_value(chrom.chromosomeID, "size_vae_hidden", sql_chrom.size_vae_hidden)
        generation.set_value(chrom.chromosomeID, "size_vae_latent", sql_chrom.size_vae_latent)
        generation.set_value(chrom.chromosomeID, "num_dnn_layers", sql_chrom.num_dnn_layers)
        generation.set_value(chrom.chromosomeID, "num_vae_layers", sql_chrom.num_vae_layers)
        generation.set_value(chrom.chromosomeID, "val_vae_reconstruction_loss", sql_chrom.val_vae_reconstruction_loss)
        generation.set_value(chrom.chromosomeID, "val_vae_latent_args_loss", sql_chrom.val_vae_latent_args_loss)
        generation.set_value(chrom.chromosomeID, "val_dnn_latent_layer_loss", sql_chrom.val_dnn_latent_layer_loss)
        generation.set_value(chrom.chromosomeID, "val_dnn_predictor_layer_loss", sql_chrom.val_dnn_predictor_layer_loss)
        generation.set_value(chrom.chromosomeID, "hostname", sql_chrom.hostname)
        generation.set_value(chrom.chromosomeID, "start_time", sql_chrom.start_time)
        generation.set_value(chrom.chromosomeID, "end_time", sql_chrom.end_time)
        generation.set_value(chrom.chromosomeID, "run_time", sql_chrom.run_time)
        generation.set_value(chrom.chromosomeID, "num_generations", sql_chrom.num_generations)
        generation.set_value(chrom.chromosomeID, "population_size", sql_chrom.population_size)
        generation.set_value(chrom.chromosomeID, "mutate_prob", sql_chrom.mutate_prob)
        generation.set_value(chrom.chromosomeID, "cross_prob", sql_chrom.cross_prob)
        generation.set_value(chrom.chromosomeID, "train_file", sql_chrom.train_file)
        generation.set_value(chrom.chromosomeID, "table_dir", sql_chrom.table_dir)
        generation.set_value(chrom.chromosomeID, "model_dir", sql_chrom.model_dir)
        generation.set_value(chrom.chromosomeID, "log_dir", sql_chrom.log_dir)
        generation.set_value(chrom.chromosomeID, "dnn_log_var_prior", sql_chrom.dnn_log_var_prior)
        generation.set_value(chrom.chromosomeID, "w_kl_anneal", sql_chrom.w_kl_anneal)
        generation.set_value(chrom.chromosomeID, "kl_anneal", sql_chrom.kl_anneal)
        generation.set_value(chrom.chromosomeID, "patience", sql_chrom.patience)
        generation.set_value(chrom.chromosomeID, "prediction_log_var_prior", sql_chrom.prediction_log_var_prior)
        generation.set_value(chrom.chromosomeID, "dnn_kl_weight", sql_chrom.dnn_kl_weight)
        generation.set_value(chrom.chromosomeID, "vae_kl_weight", sql_chrom.vae_kl_weight)
        generation.set_value(chrom.chromosomeID, "vae_weight", sql_chrom.vae_weight)
        generation.set_value(chrom.chromosomeID, "dnn_weight", sql_chrom.dnn_weight)
        generation.set_value(chrom.chromosomeID, "num_epochs", sql_chrom.num_epochs)
        generation.set_value(chrom.chromosomeID, "optimizer", sql_chrom.optimizer)
        generation.set_value(chrom.chromosomeID, "batch_size", sql_chrom.batch_size)
        generation.set_value(chrom.chromosomeID, "predictor_type", sql_chrom.predictor_type)
        generation.set_value(chrom.chromosomeID, "run_name", sql_chrom.run_name)
        generation.set_value(chrom.chromosomeID, "isTrained", sql_chrom.isTrained)
        generation.set_value(chrom.chromosomeID, "fitness", sql_chrom.fitness)

    return generation

def select_parents(generation):
    '''Generate two random numbers between 0 and total_fitness
        not including total_fitness'''

    a = 0.5
    c = 40
    b = -10
    d = 10
    total_fitness = sum(np.exp(generation.fitness*a + b)*c+d)
    assert(total_fitness >= 0), '`total_fitness` should not be negative'

    rand_parent1 = random.random()*total_fitness
    rand_parent2 = random.random()*total_fitness

    parent1 = None
    parent2 = None

    fitness_count = 0
    for chromosome in generation.itertuples():
        #fitness_count += chromosome.fitness
        fitness_count += np.exp(chromosome.fitness*a + b)*c+d

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
            if(param in ["size_kernel", "size_pool", "size_filter"]):
                continue
            p1_param = generation.ix[idx_parent1, param]
            p2_param = generation.ix[idx_parent2, param]

            child_gene = random.choice([p1_param, p2_param])
            new_generation.set_value(chromosomeID, param, child_gene)

        #Merge Kernel sizes, Max Pooling layers and Filter sizes
        num_conv_layers = new_generation.ix[chromosomeID, "num_conv_layers"]
        parent_choises = np.random.choice([True, False], size = num_conv_layers) #Boolean array to decide whether to choose from parent1 or parent2

        for array_param in ["size_kernel", "size_pool", "size_filter"]:
            new_array = np.array([0]*num_conv_layers, dtype=int)
            for i in range(num_conv_layers):
                array_p1 = generation.ix[idx_parent1, array_param]
                array_p2 = generation.ix[idx_parent2, array_param]
                if(len(array_p1) <= i):
                    new_array[i] = array_p2[i]
                elif(len(array_p2) <= i):
                    new_array[i] = array_p1[i]
                else:
                    new_array[i] = array_p1[i] if parent_choises[i] else array_p2[i]
            new_generation.set_value(chromosomeID, array_param, new_array)

        new_generation.set_value(chromosomeID, 'info', 'Child of '+str(parent1.chromosomeID)+' and '+str(parent2.chromosomeID))
    else:
        crossover_happened = False

        p1_fitness = generation.ix[idx_parent1, 'fitness']
        p2_fitness = generation.ix[idx_parent2, 'fitness']

        idx_child = idx_parent1 if p1_fitness > p2_fitness else idx_parent1
        params_copy = param_choices + ['fitness', 'hostname', 'start_time', 'end_time', 'run_time', 'isTrained',
                                       'val_vae_reconstruction_loss', 'val_vae_latent_args_loss', 'val_dnn_latent_layer_loss', 'val_dnn_predictor_layer_loss']
        parent_p = generation.loc[idx_child, params_copy]
        new_generation.set_value(chromosomeID, params_copy, parent_p)
        new_generation.set_value(chromosomeID, 'info', 'Descendant of '+str(generation.loc[idx_child, 'chromosomeID']))

    return crossover_happened

def mutate(new_generation, chromosomeID,
            prob, param_choices, verbose = False):

    if verbose:
        print('Mutating Child {}'.format(chromosomeID))

    mutation_happened = False
    for param, (range_change, min_val) in param_choices.items():
        if(random.random() <= prob):
            mutation_happened = True

            if(param not in ["num_conv_layers", "size_kernel", "size_pool", "size_filter"]):
                # Compute delta_param step
                change_p = np.random.uniform(-range_change, range_change)

                # Add delta_param to param
                current_p = new_generation.loc[chromosomeID, param] + change_p

                # If param less than `min_val`, then set param to `min_val`
                current_p = np.max([current_p, min_val])
                current_p = np.int(np.round(current_p))

                # All params must be integer sized: round and convert
                new_generation.set_value(chromosomeID, param, current_p)
            elif param == "num_conv_layers":
                # Add or remove 1 Convolution layer

                num_conv_layers = new_generation.loc[chromosomeID, "num_conv_layers"]
                size_kernel = new_generation.loc[chromosomeID, "size_kernel"]
                size_pool = new_generation.loc[chromosomeID, "size_pool"]
                size_filter = new_generation.loc[chromosomeID, "size_filter"]

                change_p = np.int(np.round(np.random.uniform(-1, 1)))
                if(change_p == -1):
                    if(num_conv_layers > 0):
                        del_index = random.choice(range(num_conv_layers))
                        size_kernel = np.delete(size_kernel, del_index)
                        size_pool = np.delete(size_pool, del_index)
                        size_filter = np.delete(size_filter, del_index)
                        num_conv_layers -= 1
                elif(change_p == 1):
                    if(num_conv_layers > 0):
                        dup_index = random.choice(range(num_conv_layers))
                        size_kernel = np.append(size_kernel, size_kernel[dup_index])
                        size_pool = np.append(size_pool, size_pool[dup_index])
                        size_filter = np.append(size_filter, size_filter[dup_index])
                    else:
                        size_kernel = np.append(size_kernel, 0)
                        size_pool = np.append(size_pool, 0)
                        size_filter = np.append(size_filter, 0)
                    num_conv_layers += 1

                new_generation.set_value(chromosomeID, "num_conv_layers", num_conv_layers)
                new_generation.set_value(chromosomeID, "size_kernel", size_kernel)
                new_generation.set_value(chromosomeID, "size_pool", size_pool)
                new_generation.set_value(chromosomeID, "size_filter", size_filter)

            elif param in ["size_kernel", "size_pool", "size_filter"]:
                # Choose a random index and mutate it

                num_conv_layers = new_generation.loc[chromosomeID, "num_conv_layers"]
                array_p = new_generation.loc[chromosomeID, param]
                if(num_conv_layers > 0):
                    change_index = random.choice(range(num_conv_layers))

                    change_p = np.random.uniform(-range_change, range_change)
                    current_p = array_p[change_index] + change_p
                    current_p = np.max([current_p, min_val])
                    current_p = np.int(np.round(current_p))

                    array_p[change_index] = current_p
                    new_generation.set_value(chromosomeID, param, array_p)

    if(mutation_happened):
        new_generation.set_value(chromosomeID, 'info', new_generation.loc[chromosomeID, 'info']+" [Mutated]")

    return mutation_happened

def activate_workers(logdir = 'train_logs',
                    git_dir = 'vaelstmpredictor',
                    verbose = True):
    machines = [{"host": "172.16.50.181", "username": "acc",
                    "key_filename": key_filename},
                #{"host": "172.16.50.176", "username": "acc",
                #    "key_filename": key_filename},
                {"host": "172.16.50.177", "username": "acc",
                    "key_filename": key_filename},
                #{"host": "172.16.50.163", "username": "acc",
                #    "key_filename": key_filename},
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
                    "key_filename": key_filename}]
    for machine in machines:
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(machine["host"], key_filename=machine['key_filename'])
        except NoValidConnectionsError as error:
            warning_message(error)
            ssh.close()

        stdin, stdout, stderr = ssh.exec_command('ls | grep {}'.format(git_dir))
        if(len(stdout.readlines()) == 0):
            git_clone(machine["host"], key_filename=machine['key_filename'])
        elif verbose:
            info_message('File {} exists on {}'.format(git_dir, machine['host']))

        stdin, stdout, stderr = ssh.exec_command('nohup python {}/RunWorker.py'.format(git_dir))

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
