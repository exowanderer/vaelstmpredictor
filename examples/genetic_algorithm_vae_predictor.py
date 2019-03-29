# from https://github.com/philippesaade11/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --iterations 500 --population_size 10 --num_epochs 200
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from contextlib import redirect_stdout

from keras import backend as K
from keras.utils import to_categorical

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
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

def generate_random_chromosomes(population_size, clargs, data_instance, 
                            start_small = True, init_same = True,
                            input_size = None, vae_kl_weight = 1.0, 
                            predictor_weight = 1.0, predictor_kl_weight = 1.0, 
                            min_vae_hidden1 = 2, min_vae_latent = 2, 
                            min_dnn_hidden1 = 2, max_vae_hidden = 1024, 
                            max_vae_latent = 1024, max_dnn_hidden = 1024, 
                            verbose=False):
    # explicit defaults
    zero = 0

    # num_vae_hidden = num_vae_hidden or random.randint(1, 10)
    if input_size is not None and init_same:
        size_vae_hidden1 = random.randint(input_size//2, input_size)
    else:
        size_vae_hidden1 = random.randint(min_vae_hidden1, max_vae_hidden)
    # set to zero or random
    if start_small:
        size_vae_hidden2 = zero
        size_vae_hidden3 = zero
        size_vae_hidden4 = zero
        size_vae_hidden5 = zero
    else:
        size_vae_hidden2 = random.randint(zero, max_vae_hidden)
        size_vae_hidden3 = random.randint(zero, max_vae_hidden)
        size_vae_hidden4 = random.randint(zero, max_vae_hidden)
        size_vae_hidden5 = random.randint(zero, max_vae_hidden)

    # set to zero or random
    size_vae_latent = random.randint(min_vae_latent, max_vae_latent)
    
    # set to zero or random
    size_dnn_hidden1 = random.randint(min_dnn_hidden1, max_dnn_hidden)
    if start_small:
        size_dnn_hidden2 = zero
        size_dnn_hidden3 = zero
        size_dnn_hidden4 = zero
        size_dnn_hidden5 = zero
    else:
        size_dnn_hidden2 = random.randint(zero, max_dnn_hidden)
        size_dnn_hidden3 = random.randint(zero, max_dnn_hidden)
        size_dnn_hidden4 = random.randint(zero, max_dnn_hidden)
        size_dnn_hidden5 = random.randint(zero, max_dnn_hidden)

    generationID = 0
    generation_0 = []
    for chromosomeID in range(population_size):
        params_dict =  { 'clargs':clargs, 
                        'verbose':verbose, 
                        'data_instance':data_instance, 
                        'generationID':generationID, 
                        'chromosomeID':chromosomeID,
                        'vae_kl_weight':vae_kl_weight, 
                        'predictor_weight':predictor_weight,
                        'predictor_kl_weight':predictor_kl_weight, 
                        'size_vae_hidden1':size_vae_hidden1,
                        'size_vae_hidden2':size_vae_hidden2,
                        'size_vae_hidden3':size_vae_hidden3,
                        'size_vae_hidden4':size_vae_hidden4,
                        'size_vae_hidden5':size_vae_hidden5,
                        'size_vae_latent':size_vae_latent,
                        'size_dnn_hidden1':size_dnn_hidden1,
                        'size_dnn_hidden2':size_dnn_hidden2,
                        'size_dnn_hidden3':size_dnn_hidden3,
                        'size_dnn_hidden4':size_dnn_hidden4,
                        'size_dnn_hidden5':size_dnn_hidden5
                       }

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

def cross_over(parent1, parent2, prob, verbose=False):
    if verbose:
        print('Crossing over with Parent {} and Parent {}'.format(
                                parent1.chromosomeID,parent2.chromosomeID))
    
    if(random.random() <= prob):
        params1 = {}
        params2 = {}
        for param in Chromosome.params:
            if(random.random() <= 0.5):
                params1[param] = parent1.params_dict[param]
                params2[param] = parent2.params_dict[param]
            else:
                params1[param] = parent2.params_dict[param]
                params2[param] = parent1.params_dict[param]
        
        clargs = parent1.clargs
        data_instance = parent1.data_instance
        generationID = parent1.generationID + 1
        chromosomeID = parent1.chromosomeID
        vae_kl_weight = parent1.vae_kl_weight
        predictor_weight = parent1.predictor_weight
        predictor_kl_weight = parent1.predictor_kl_weight

        child1 = Chromosome(clargs=clargs, data_instance=data_instance, 
            generationID=generationID, chromosomeID=chromosomeID,
            vae_kl_weight = vae_kl_weight, predictor_weight = predictor_weight,
            predictor_kl_weight = predictor_kl_weight, verbose=verbose, **params1)

        child2 = Chromosome(clargs=clargs, data_instance=data_instance, 
            generationID=generationID, chromosomeID=chromosomeID,
            vae_kl_weight = vae_kl_weight, predictor_weight = predictor_weight,
            predictor_kl_weight = predictor_kl_weight, verbose=verbose, **params2)

        return child1, child2
    
    return parent1, parent2

def mutate(child, prob, range_change = 25, forced_evolve = False, 
            min_layer_size = 2, verbose = False):
    
    # explicit declaration
    zero = 0 

    if verbose:
        print('Mutating Child {} in {}'.format(child.chromosomeID, 
                                                child.generationID))
    
    for param in Chromosome.params:
        if(random.random() <= prob):
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

    return child

class Chromosome(VAEPredictor):
    
    #[number of hidden layers in VAE,
    #   size of the first hidden layer in VAE,
    #   size of the latent layer,
    #   number of hidden layers in the DNN regressor,
    #   size of the first hidden layer in the DNN regressor]
    params = ["size_vae_hidden1", "size_vae_hidden2", "size_vae_hidden3", 
                "size_vae_hidden4", "size_vae_hidden5", 
              "size_vae_latent", 
              "size_dnn_hidden1", "size_dnn_hidden2", "size_dnn_hidden3", 
                "size_dnn_hidden4", "size_dnn_hidden5"]
    
    def __init__(self, clargs, data_instance, 
                generationID = 0, chromosomeID = 0, 
                size_vae_hidden1 = 2, size_vae_hidden2 = 0, 
                size_vae_hidden3 = 0, size_vae_hidden4 = 0, 
                size_vae_hidden5 = 0, size_vae_latent = 2, 
                size_dnn_hidden1 = 2, size_dnn_hidden2 = 0, 
                size_dnn_hidden3 = 0, size_dnn_hidden4 = 0, 
                size_dnn_hidden5 = 0, vae_kl_weight = 1.0, 
                predictor_weight = 1.0, predictor_kl_weight = 1.0, 
                verbose = False):

        self.verbose = verbose
        self.clargs = clargs
        self.data_instance = data_instance
        self.generationID = generationID
        self.chromosomeID = chromosomeID
        self.time_stamp = clargs.time_stamp

        self.vae_kl_weight = vae_kl_weight
        self.predictor_weight = predictor_weight
        self.predictor_kl_weight = predictor_kl_weight

        self.params_dict =  {'size_vae_hidden1':size_vae_hidden1,
                             'size_vae_hidden2':size_vae_hidden2,
                             'size_vae_hidden3':size_vae_hidden3,
                             'size_vae_hidden4':size_vae_hidden4,
                             'size_vae_hidden5':size_vae_hidden5,
                             'size_vae_latent':size_vae_latent,
                             'size_dnn_hidden1':size_dnn_hidden1,
                             'size_dnn_hidden2':size_dnn_hidden2,
                             'size_dnn_hidden3':size_dnn_hidden3,
                             'size_dnn_hidden4':size_dnn_hidden4,
                             'size_dnn_hidden5':size_dnn_hidden5
                            }

        self.model_dir = clargs.model_dir
        self.run_name = clargs.run_name
        self.predictor_type = clargs.predictor_type
        self.original_dim = clargs.original_dim
        self.predictor_weight = clargs.predictor_weight
        
        self.optimizer = clargs.optimizer
        self.batch_size = clargs.batch_size
        self.use_prev_input = False
        self.predictor_out_dim = clargs.n_labels

        self.vae_hidden_dims = [size_vae_hidden1, size_vae_hidden2, 
                                 size_vae_hidden3, size_vae_hidden4, 
                                 size_vae_hidden5]

        self.vae_latent_dim = size_vae_latent

        self.predictor_hidden_dims = [size_dnn_hidden1, size_dnn_hidden2, 
                                       size_dnn_hidden3, size_dnn_hidden4, 
                                       size_dnn_hidden5]
        self.predictor_latent_dim = clargs.n_labels-1
        
        self.get_model()
        self.neural_net = self.model
        self.fitness = 0
        
        assert(os.path.exists(self.model_dir)), "{} does not exist.".format(self.model_dir) 
        self.model_topology_savefile = '{}/{}_{}_{}_model_topology_savefile_{}.save'
        self.model_topology_savefile = self.model_topology_savefile.format(self.model_dir, self.run_name,self.generationID, self.chromosomeID,
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

        if verbose: print(self.neural_net.summary())

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
            self.predictor_kl_weight = K.variable(value=0.0)
        
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
        
        self.best_val_loss = sum([val for key,val in self.best_loss.items() if 'val_' in key])
        self.fitness = 1.0 / self.best_val_loss

        if verbose: 
            print("Generation: {}".format(self.generationID))
            print("Chromosome: {}".format(self.chromosomeID))
            print("Operation Time: {}".format(time() - start_train))
            print('\nBest Loss:')
            for key,val in self.best_loss.items():
                print('{}: {}'.format(key,val))

            print('\nFitness: {}'.format(self.fitness))
        
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
        joblib.dump({'best_loss':self.best_loss,'history':self.history}, 
                        joblib_save_loc)

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
    parser.add_argument('--predictor_weight', type=float, default=1.0,
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
    parser.add_argument('--log_dir', type=str, default='data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str, default='data/models',
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

    clargs = parser.parse_args()
    
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

    generation = generate_random_chromosomes(population_size,
                    clargs = clargs, data_instance = data_instance,
                    start_small = True, verbose = verbose)

    # generationID = 0    
    # generation_dict = {}
    # generation_dict['params'] = member.params_dict
    # generation_dict['fitness'] = member.fitness
    # evolutionary_tree = {generationID:generation_dict}

    best_fitness = []
    fig = plt.gcf()
    fig.show()

    start = time()
    # while gen_num < iterations:
    for _ in range(iterations):
        start_while = time()
        #Create new generation
        new_generation = []
        # gen_num += 1
        for _ in range(population_size//2):
            parent1, parent2 = select_parents(generation)
            child1, child2 = cross_over(parent1, parent2, cross_prob, 
                                        verbose=verbose)
            
            mutate(child1, mutate_prob, verbose=verbose)
            mutate(child2, mutate_prob, verbose=verbose)
            
            # gene_set1 = child1.params_dict
            # gene_set2 = child2.params_dict
            
            # if not (gene_set1 == evolutionary_tree).all(axis=1).any():
            child1.train()

            # if not (gene_set2 == evolutionary_tree).all(axis=1).any():
            child2.train()
            
            new_generation.append(child1)
            new_generation.append(child2)

        print('Time for Generation{}: {} minutes'.format(child1.generationID, 
                                                    (time() - start_while)//60))

        generation = new_generation
        
        # del new_generation.data_instance
        # del new_generation.neural_net
        # del new_generation.model

        # evolutionary_tree.append(new_generation)
        
        best_fitness.append(max(chrom.fitness for chrom in generation))
        
        if make_plots:
            plt.plot(best_fitness, color="c")
            plt.xlim([0, iterations])
            fig.canvas.draw()