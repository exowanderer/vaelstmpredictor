import sys, os.path
sys.path.append(os.path.abspath('../'))

import numpy as np
import socket
from time import time
from tqdm import tqdm
import argparse
from database import Variables, Chromosome, db

from GeneticAlgorithm_Conv1D import generate_random_chromosomes, train_generation, create_blank_dataframe, select_parents, cross_over, mutate, load_generation_from_sql, refactor_weights

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
    parser.add_argument('--max_vae_latent', type=int, default=5,
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--max_vae_hidden', type=int, default=512,
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--max_dnn_hidden', type=int, default=512,
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--max_dnn_hidden_layers', type=int, default=5,
                help='Maximum number of DNN hidden layers')
    parser.add_argument('--max_conv_layers', type=int, default=5,
                help='Maximum number of Convolution layers')
    parser.add_argument('--max_kernel_size', type=int, default=6,
                help='Maximum kernel size (x2 +1)')
    parser.add_argument('--max_pool_size', type=int, default=4,
                help='Maximum Max Pooling size (x2)')
    parser.add_argument('--max_filter_size', type=int, default=128,
                help='Maximum number of filters for each convolution layer')
    parser.add_argument('--min_vae_hidden_layers', type=int, default=1,
                help='minimum number of VAE hidden layers')
    parser.add_argument('--min_vae_latent', type=int, default=1,
                help='minimum number of VAE neurons per layer')
    parser.add_argument('--min_vae_hidden', type=int, default=2,
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--min_dnn_hidden', type=int, default=2,
                help='Maximum number of VAE neurons per layer')
    parser.add_argument('--min_dnn_hidden_layers', type=int, default=1,
                help='minimum number of DNN hidden layers')
    parser.add_argument('--min_conv_layers', type=int, default=1,
                help='Minimum number of Convolution layers')
    parser.add_argument('--min_kernel_size', type=int, default=0,
                help='Minimum kernel size (x2 +1)')
    parser.add_argument('--min_pool_size', type=int, default=0,
                help='Minimum Max Pooling size (x2)')
    parser.add_argument('--min_filter_size', type=int, default=1,
                help='Minimum number of filters for each convolution layer')
    parser.add_argument('--dnn_weight', type=float, default=1,
                help='relative weight on prediction loss')
    parser.add_argument('--vae_weight', type=float, default=1,
                help='relative weight on prediction loss')
    parser.add_argument('--vae_kl_weight', type=float, default=1,
                help='relative weight on prediction loss')
    parser.add_argument('--dnn_kl_weight', type=float, default=1,
                help='relative weight on prediction loss')
    parser.add_argument('--prediction_log_var_prior', type=float, default=0.0,
                help='w log var prior')
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
    parser.add_argument('--train_file', type=str, default='exoplanet',
                help='file of training data (.pickle)')
    parser.add_argument('--cross_prob', type=float, default=0.7,
                help='Probability of crossover between generations')
    parser.add_argument('--mutate_prob', type=float, default=0.01,
                help='Probability of mutation for each member')
    parser.add_argument('--population_size', type=int, default=10,
                help='size of the population to evolve; '\
                        'preferably divisible by 2')
    parser.add_argument('--num_generations', type=int, default=10,
                help='number of generations for genetic algorithm')
    parser.add_argument('--verbose', action='store_true',
                help='print more [INFO] and [DEBUG] statements')
    parser.add_argument('--sshport', type=int, default=22,
                help='IP port over which to ssh')
    parser.add_argument('--sleep_time', type=float, default=10,
            help='Time to pause in between repetitive sql queries')
    parser.add_argument('--send_back', action='store_true',
            help='Toggle whether to send the ckpt file + population local csv')
    parser.add_argument('--save_model', action='store_true',
            help='Save model ckpt.s and other stored values')

    clargs = parser.parse_args()

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
    sleep_time = clargs.sleep_time

    clargs.time_stamp = int(time())
    clargs.run_name = '{}_{}'.format(clargs.run_name, clargs.time_stamp)

    if verbose: print('\n\n[INFO] Run Base Name: {}\n'.format(clargs.run_name))

    #Save is Done in the database
    isDone = db.session.query(Variables).filter(Variables.name == "isDone").first()
    if isDone == None:
        var = Variables(name="isDone", value=0)
        db.session.add(var)
        db.session.commit()
    else:
        isDone.value = 0
        db.session.commit()

    #Save Generation ID in the database
    CurrentGen = db.session.query(Variables).filter(Variables.name == "CurrentGen").first()
    if CurrentGen == None:
        CurrentGen = Variables(name="CurrentGen", value=0)
        db.session.add(CurrentGen)
        db.session.commit()

    check = db.session.query(Chromosome).filter(Chromosome.generationID == CurrentGen.value).first()
    if(check == None):
        generation = generate_random_chromosomes(population_size = population_size,
                            clargs = clargs,
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
                            min_conv_layers = clargs.min_conv_layers,
                            max_conv_layers = clargs.max_conv_layers,
                            min_kernel_size = clargs.min_kernel_size,
                            max_kernel_size = clargs.max_kernel_size,
                            max_filter_size = clargs.max_filter_size,
                            min_filter_size = clargs.min_filter_size,
                            verbose = clargs.verbose)
        CurrentGen.value = 0
    else:
        info_message("Loaded Generation From DB")
        generation = load_generation_from_sql(CurrentGen.value, population_size)

    generationID = CurrentGen.value
    generation = train_generation(generation, clargs, verbose=verbose, sleep_time=sleep_time, save_DB=(check == None))

    best_fitness = []
    fitnesses = generation.fitness.values
    new_best_fitness = generation.fitness.values.max()

    if verbose:
        info_message('For Generation: {}, the best fitness was {}'.format(
                generationID, new_best_fitness))

    best_fitness.append(new_best_fitness)

    param_choices = {'num_vae_layers': (1,1),
                     'num_dnn_layers': (1,1),
                     'size_vae_latent': (10,1),
                     'size_vae_hidden': (50,1),
                     'size_dnn_hidden': (50,1),
                     'num_conv_layers': (1,1),
                     'size_kernel': (1,0),
                     'size_pool': (1,0),
                     'size_filter': (10,1)}

    start = time()
    # while gen_num < num_generations:
    for generationID in range(generationID+1,num_generations):

        #Save Generation ID in the database
        CurrentGen = db.session.query(Variables).filter(Variables.name == "CurrentGen").first()
        if CurrentGen == None:
            var = Variables(name="CurrentGen", value=generationID)
            db.session.add(var)
            db.session.commit()
        else:
            CurrentGen.value = generationID
            db.session.commit()

        start_while = time()
        # Create new generation
        new_generation = create_blank_dataframe(generationID, population_size)

        for chromosomeID in tqdm(range(population_size)):
            parent1, parent2 = select_parents(generation)
            crossover_happened = cross_over(new_generation, generation,
                                            parent1, parent2, chromosomeID,
                                            list(param_choices.keys()), cross_prob,
                                            verbose = verbose)

            mutation_happened = mutate(new_generation, chromosomeID, mutate_prob,
                                            param_choices, verbose = verbose)

            isTrained = not (mutation_happened or crossover_happened)
            if not isTrained:
                new_generation.set_value(chromosomeID, 'fitness', -1.0)

            isTrained = 2 if isTrained else 0

            new_generation.set_value(chromosomeID, 'isTrained', isTrained)
            new_generation.set_value(chromosomeID, 'generationID',generationID)
            new_generation.set_value(chromosomeID, 'chromosomeID',chromosomeID)

            info_message('Adding Chromosome:\n{}'.format(
                    new_generation.iloc[chromosomeID]))

        # Re-sort by chromosomeID
        new_generation = new_generation.sort_values('chromosomeID')
        new_generation.index = np.arange(population_size)

        refactor_weights(new_generation, generation)
        generation = train_generation(new_generation, clargs, verbose=verbose, sleep_time=sleep_time)

        info_message('Time for Generation{}: {} minutes'.format(generationID,
                                            (time() - start_while)//60))

        fitnesses = generation.fitness.values
        new_best_fitness = generation.fitness.values.max()

        if verbose:
            info_message('For Generation: {}, the best fitness was {}'.format(
                    generationID, new_best_fitness))

        best_fitness.append(new_best_fitness)

    isDone = db.session.query(Variables).filter(Variables.name == "isDone").first()
    if isDone == None:
        var = Variables(name="isDone", value=1)
        db.session.add(var)
        db.session.commit()
    else:
        isDone.value = 1
        db.session.commit()

    if verbose: print("Done.")
