# from https://github.com/philippesaade11/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --iterations 500 --population_size 10 --num_epochs 200
import argparse
# import matplotlib.pyplot as plt
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

class BlankClass(object):
    def __init__(self):
        pass

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
                        verbose=False):
    
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
        chrom.train()
        generation_0.append(chrom)
    
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

class Chromosome(VAEPredictor):
    # params = ["size_vae_hidden1", "size_vae_hidden2", "size_vae_hidden3", 
    #             "size_vae_hidden4", "size_vae_hidden5", 
    #           "vae_latent_dim", 
    #           "size_dnn_hidden1", "size_dnn_hidden2", "size_dnn_hidden3", 
    #             "size_dnn_hidden4", "size_dnn_hidden5"]

    def __init__(self, clargs, data_instance, vae_latent_dim, 
                vae_hidden_dims, dnn_hidden_dims, 
                generationID = 0, chromosomeID = 0, 
                vae_kl_weight = 1.0, dnn_weight = 1.0, 
                dnn_kl_weight = 1.0, verbose = False,
                save_as_you_train = True):

        self.verbose = verbose
        self.save_as_you_train = save_as_you_train
        self.clargs = clargs
        self.data_instance = data_instance
        self.generationID = generationID
        self.chromosomeID = chromosomeID
        self.time_stamp = clargs.time_stamp
        
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.dnn_hidden_dims = dnn_hidden_dims

        self.vae_kl_weight = vae_kl_weight
        self.dnn_weight = dnn_weight
        self.dnn_kl_weight = dnn_kl_weight
        
        self.params_dict = {}
        for k, layer_size in enumerate(self.vae_hidden_dims):
            self.params_dict['size_vae_hidden{}'.format(k)] = layer_size

        self.params_dict['vae_latent_dim'] = self.vae_latent_dim

        for k, layer_size in enumerate(self.dnn_hidden_dims):
            self.params_dict['size_dnn_hidden{}'.format(k)] = layer_size

        self.model_dir = clargs.model_dir
        self.run_name = clargs.run_name
        self.predictor_type = clargs.predictor_type
        self.original_dim = clargs.original_dim
        self.dnn_weight = clargs.dnn_weight
        
        self.optimizer = clargs.optimizer
        self.batch_size = clargs.batch_size
        self.use_prev_input = False
        self.dnn_out_dim = clargs.n_labels

        self.dnn_latent_dim = clargs.n_labels-1
        
        self.get_model()
        self.neural_net = self.model
        self.fitness = 0
        
        assert(os.path.exists(self.model_dir)), "{} does not exist.".format(self.model_dir) 
        self.model_topology_savefile = '{}/{}_{}_{}_model_topology_savefile_{}.save'
        self.model_topology_savefile = self.model_topology_savefile.format(self.model_dir, self.run_name, self.generationID, self.chromosomeID,
            self.time_stamp)

        with open(self.model_topology_savefile, 'w') as f:
            with redirect_stdout(f):
                self.neural_net.summary()

        yaml_filename = self.model_topology_savefile.replace('.save', '.yaml')
        with open(yaml_filename, 'w') as yaml_fileout:
            yaml_fileout.write(self.neural_net.to_yaml())
        
        # save model args
        json_filename = self.model_topology_savefile.replace('.save', '.json')
        with open(json_filename, 'w') as json_fileout:
            json_fileout.write(self.neural_net.to_json())

        if verbose: self.neural_net.summary()

    def train(self, verbose = False):
        """Training control operations to create VAEPredictor instance, 
            organize the input data, and train the network.
        
        Args:
            clargs (object): command line arguments from `argparse`
                Structure Contents: n_labels,
                    run_name, patience, kl_anneal, do_log, do_chkpt, num_epochs
                    w_kl_anneal, optimizer, batch_size
            
            data_instance (object): 
                Object instance for organizing data structures
                Structure Contents: train_labels, valid_labels, test_labels
                    labels_train, data_train, labels_valid, data_valid
        """
        start_train = time()
        verbose = verbose or self.verbose
        
        DI = self.data_instance

        predictor_train = to_categorical(DI.train_labels, self.clargs.n_labels)
        predictor_validation = to_categorical(DI.valid_labels,self.clargs.n_labels)

        min_epoch = max(self.clargs.kl_anneal, self.clargs.w_kl_anneal)+1
        callbacks = get_callbacks(self.clargs, patience=self.clargs.patience, 
                    min_epoch = min_epoch, do_log = self.clargs.do_log, 
                    do_chckpt = self.clargs.do_chckpt)

        if clargs.kl_anneal > 0: 
            self.vae_kl_weight = K.variable(value=0.1)
        if clargs.w_kl_anneal > 0: 
            self.dnn_kl_weight = K.variable(value=0.0)
        
        # self.clargs.optimizer, was_adam_wn = init_adam_wn(self.clargs.optimizer)
        # self.clargs.optimizer = 'adam' if was_adam_wn else self.clargs.optimizer
        
        save_model_in_pieces(self.model, self.clargs)
        
        vae_train = DI.data_train
        vae_features_val = DI.data_valid

        data_based_init(self.model, DI.data_train[:clargs.batch_size])

        vae_labels_val = [DI.labels_valid, predictor_validation, 
                            predictor_validation,DI.labels_valid]

        validation_data = (vae_features_val, vae_labels_val)
        train_labels = [DI.labels_train, predictor_train, predictor_train, DI.labels_train]
        
        self.compile()

        self.history = self.model.fit(vae_train, train_labels,
                                    shuffle = True,
                                    epochs = clargs.num_epochs,
                                    batch_size = clargs.batch_size,
                                    callbacks = callbacks,
                                    validation_data = validation_data)

        max_kl_anneal = max(clargs.kl_anneal, clargs.w_kl_anneal)
        self.best_ind = np.argmin([x if i >= max_kl_anneal + 1 else np.inf \
                    for i,x in enumerate(self.history.history['val_loss'])])
        
        self.best_loss = {k: self.history.history[k][self.best_ind] \
                                        for k in self.history.history}
        
        # self.best_val_loss = sum([val for key,val in self.best_loss.items() \
        #                             if 'val_' in key and 'loss' in key])
        
        self.fitness = 1.0 / best_loss['val_loss']
        
        if verbose: 
            print("Generation: {}".format(self.generationID))
            print("Chromosome: {}".format(self.chromosomeID))
            print("Operation Time: {}".format(time() - start_train))
            print('\nBest Loss:')
            for key,val in self.best_loss.items():
                print('{}: {}'.format(key,val))

            print('\nFitness: {}'.format(self.fitness))
        
        if self.save_as_you_train: self.save()

    def save(self):
        joblib_save_loc = '{}/{}_{}_{}_trained_model_output_{}.joblib.save'
        joblib_save_loc = joblib_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)

        wghts_save_loc = '{}/{}_{}_{}_trained_model_weights_{}.save'
        wghts_save_loc = wghts_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)
        
        model_save_loc = '{}/{}_{}_{}_trained_model_full_{}.save'
        model_save_loc = model_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)
        
        self.neural_net.save_weights(wghts_save_loc, overwrite=True)
        self.neural_net.save(model_save_loc, overwrite=True)

        try:
            joblib.dump({'best_loss':self.best_loss,'history':self.history}, 
                        joblib_save_loc)
        except Exception as e:
            print(str(e))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, default='ga_test_',
               help='tag for current run')
    parser.add_argument('--predictor_type', type=str, default="classification",
                help='select `classification` or `regression`')
    parser.add_argument('--batch_size', type=int, default=128,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                help='optimizer name') 
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--start_small', action='store_true',
                help='Only the first hidden layer is initially populated')
    parser.add_argument('--init_large', action='store_true', 
                help='Initial the 1st layer in [num_features/2,num_features]')
    parser.add_argument('--max_vae_hidden_layers', type=int, default=5, 
                help='Maximum number of VAE hidden layers')
    parser.add_argument('--max_vae_latent', type=int, default=1024, 
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--max_dnn_latent', type=int, default=1024, 
                help='Maximum number of DNN neurons per layer')
    parser.add_argument('--max_dnn_hidden_layers', type=int, default=5,
                help='Maximum number of DNN hidden layers')
    parser.add_argument('--dnn_weight', type=float, default=1.0,
                help='relative weight on prediction loss')
    parser.add_argument('--prediction_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--do_chckpt", action="store_true",
                help="save model checkpoints")
    parser.add_argument('--patience', type=int, default=10,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='../data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str, default='../data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str, default='MNIST',
                help='file of training data (.pickle)')
    parser.add_argument('--cross_prob', type=float, default=0.7,
                help='Probability of crossover between generations')
    parser.add_argument('--mutate_prob', type=float, default=0.01,
                help='Probability of mutation for each member')
    parser.add_argument('--population_size', type=int, default=10,
                help='size of the population to evolve; '\
                        'preferably divisible by 2')
    parser.add_argument('--iterations', type=int, default=50,
                help='number of iterations for genetic algorithm')
    parser.add_argument('--verbose', action='store_true',
                help='print more [INFO] and [DEBUG] statements')
    parser.add_argument('--make_plots', action='store_true',
                help='make plots of growth in the best_loss over generations')
    parser.add_argument('-g', '--n_gpus', type=int, default=1,
                help='Input the number of GPUs to use for this operation.')
    
    clargs = parser.parse_args()
    
    if not os.path.exists(clargs.model_dir): os.mkdir(clargs.model_dir)
    if not os.path.exists(clargs.log_dir): os.mkdir(clargs.log_dir)
    
    # run_name = 'ga_test_mutli_gpus'
    # clargs.run_name = run_name
    run_name = clargs.run_name
    num_epochs = clargs.num_epochs
    cross_prob = clargs.cross_prob
    mutate_prob = clargs.mutate_prob
    population_size = clargs.population_size
    iterations = clargs.iterations
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
    
    generation = generate_random_chromosomes(
                        population_size = population_size, 
                        clargs = clargs, 
                        data_instance = data_instance)

    generationID = 0    
    evolutionary_tree = {}
    evolutionary_tree[generationID] = save_generation_to_tree(generation,   
                                                            verbose=verbose)
    
    best_fitness = []
    if make_plots:
        fig = plt.gcf()
        fig.show()

    start = time()
    # while gen_num < iterations:
    for _ in range(iterations):
        start_while = time()

        # Create new generation
        generationID += 1
        new_generation = []
        chromosomeID = 0
        for _ in range(population_size//2):
            parent1, parent2 = select_parents(generation)
            child1, child2, crossover_happened = cross_over(parent1, parent2, 
                                                cross_prob, verbose=verbose)
            
            child1.generationID = generationID
            child1.chromosomeID = chromosomeID; chromosomeID += 1 
            child2.generationID = generationID
            child2.chromosomeID = chromosomeID; chromosomeID += 1 
            
            child1, mutation_happened1 = mutate(child1, mutate_prob, 
                                                verbose=verbose)
            child1, mutation_happened2 = mutate(child2, mutate_prob, 
                                                verbose=verbose)
            
            if crossover_happened or mutation_happened1: 
                child1.train()
                new_generation.append(child1)
            else:
                new_generation.append(parent1)

            if crossover_happened or mutation_happened2: 
                child2.train()
                new_generation.append(child2)
            else:
                new_generation.append(parent2)

        print('Time for Generation{}: {} minutes'.format(child1.generationID, 
                                                (time() - start_while)//60))

        generation = new_generation
        evolutionary_tree[generationID] = save_generation_to_tree(generation,
                                                            verbose=verbose)

        best_fitness.append(max(chrom.fitness for chrom in generation))
        
        if make_plots:
            plt.plot(best_fitness, color="c")
            plt.xlim([0, iterations])
            fig.canvas.draw()

    evtree_save_name = 'evolutionary_tree_{}_ps{}_iter{}_epochs{}_cp{}_mp{}'
    evtree_save_name = evtree_save_name + '.joblib.save'
    evtree_save_name = evtree_save_name.format(run_name, population_size, 
                                iterations, num_epochs, cross_prob,mutate_prob)
    evtree_save_name = os.path.join(clargs.model_dir, evtree_save_name)

    print('[INFO] Saving evolutionary tree to {}'.format(evtree_save_name))
    joblib.dump(evolutionary_tree, evtree_save_name)