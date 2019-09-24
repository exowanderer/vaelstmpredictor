import argparse
import json
import numpy as np
import os
import requests
import socket

from vaelstmpredictor.Chromosome_DNN import Chromosome

from time import time, sleep
from keras import backend as K


def debug_message(message):
    print('[DEBUG] {}'.format(message))


def warning_message(message):
    print('[WARNING] {}'.format(message))


def info_message(message):
    print('[INFO] {}'.format(message))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_server',
                        default='LAUDeepGenerativeGenetics.pythonanywhere.com',
                        help='The URL or IP of the SQL server')
    clargs = parser.parse_args()

    for key, val in clargs.__dict__.items():
        if 'dir' in key:
            if not os.path.exists(val):
                os.mkdir(val)

    base_url = clargs.sql_server

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    hostname = s.getsockname()[0]
    s.close()

    while True:
        try:
            info_message('Grabbing Untrained Chromosome')
            chromosome = requests.get(
                url="http://{}/GetUnTrainedChrom".format(base_url))
        except:
            chromosome = None

        if(chromosome != None and chromosome.text == "0"):
            info_message("No more Chromosomes to train")
            sleep(30)
        elif(chromosome != None):
            try:
                info_message('Accessing Params from Untrained Chromosome')
                params = chromosome.json()
            except:
                continue

            clargs.batch_size = params["batch_size"]
            clargs.chromosomeID = params["chromosomeID"]
            clargs.cross_prob = params["cross_prob"]
            clargs.dnn_kl_weight = params["dnn_kl_weight"]
            clargs.dnn_log_var_prior = params["dnn_log_var_prior"]
            clargs.dnn_weight = params["dnn_weight"]
            clargs.fitness = params["fitness"]
            clargs.generationID = params["generationID"]
            clargs.isTrained = params["isTrained"]
            clargs.kl_anneal = params["kl_anneal"]
            clargs.log_dir = params["log_dir"]
            clargs.do_log = True
            clargs.do_ckpt = False
            clargs.verbose = True
            clargs.save_model = False
            #clargs.model_dir = params["model_dir"]
            clargs.model_dir = "data/models"
            clargs.mutate_prob = params["mutate_prob"]
            clargs.num_epochs = params["num_epochs"]
            clargs.num_generations = params["num_generations"]
            clargs.optimizer = params["optimizer"]
            clargs.patience = params["patience"]
            clargs.population_size = params["population_size"]
            clargs.prediction_log_var_prior = params["prediction_log_var_prior"]
            clargs.predictor_type = 'regression'  # params["predictor_type"]
            clargs.run_name = params["run_name"]
            clargs.table_dir = params["table_dir"]
            clargs.train_file = params["train_file"]
            clargs.w_kl_anneal = params["w_kl_anneal"]

            clargs.size_dnn_hidden = params["size_dnn_hidden"]
            clargs.num_dnn_layers = params["num_dnn_layers"]
            clargs.num_conv_layers = params["num_conv_layers"]
            clargs.size_kernel = params["size_kernel"]
            clargs.size_pool = params["size_pool"]
            clargs.size_filter = params["size_filter"]

            clargs.hostname = hostname

            dnn_hidden_dims = [clargs.size_dnn_hidden] * clargs.num_dnn_layers
            size_kernel = np.array(json.loads(clargs.size_kernel))*2 +1
            size_pool = np.array(json.loads(clargs.size_pool))*2
            size_filter = np.array(json.loads(clargs.size_filter))

            chrom_params = {}
            chrom_params['batch_size'] = clargs.batch_size
            chrom_params['optimizer'] = clargs.optimizer
            chrom_params['num_epochs'] = clargs.num_epochs
            chrom_params['train_file'] = clargs.train_file
            chrom_params['save_model'] = clargs.save_model
            chrom_params['verbose'] = clargs.verbose
            chrom_params['log_dir'] = clargs.log_dir
            chrom_params['model_dir'] = clargs.model_dir
            chrom_params['table_dir'] = clargs.table_dir

            chrom_params['dnn_weight'] = clargs.dnn_weight
            chrom_params['dnn_kl_weight'] = clargs.dnn_kl_weight

            chrom_params['num_conv_layers'] = clargs.num_conv_layers
            chrom_params['dnn_hidden_dims'] = dnn_hidden_dims
            chrom_params['size_kernel'] = size_kernel
            chrom_params['size_pool'] = size_pool
            chrom_params['size_filter'] = size_filter

            info_message('\n\nParams for this VAE_NN:')
            for key, val in clargs.__dict__.items():
                print('{:20}{}'.format(key, val))

            info_message("Training Chromosome " + str(clargs.chromosomeID) +
                         " Generation " + str(clargs.generationID))
            try:
                K.clear_session()
                chromosome = Chromosome(**chrom_params)

                start_time = time()

                if clargs.verbose:
                    info_message('Start Training: {}'.format(start_time))

                chromosome.train()
                K.clear_session()

                end_time = time()
                run_time = end_time - start_time

                if clargs.verbose:
                    info_message('End Training: {}'.format(end_time))
                    info_message('Runtime Training: {}'.format(run_time))

            except Exception as e:
                warning_message("Error has occured while training")
                warning_message(e)
                chromosome = None
                continue

            info_message('\n')
            print('Result: ', end=" ")
            print('GenerationID: {}'.format(clargs.generationID), end=" ")
            print('ChromosomeID: {}'.format(clargs.chromosomeID), end=" ")
            print('Fitness: {}\n'.format(chromosome.fitness))

            params["isTrained"] = 2
            params["fitness"] = chromosome.fitness
            params["val_dnn_latent_args_loss"] = chromosome.best_losses['val_dnn_latent_layer_loss']
            params["val_dnn_predictor_layer_loss"] = chromosome.best_losses['val_dnn_predictor_layer_loss']

            params["hostname"] = clargs.hostname
            params["start_time"] = start_time
            params["end_time"] = end_time
            params["run_time"] = run_time

            sent = False
            while(not sent):
                try:
                    resp = requests.get(
                        url="http://{}/AddChrom".format(base_url),
                        params=params)
                    info_message("Response: " + resp.text)
                    sent = (resp.text == "1")
                except:
                    sent = False