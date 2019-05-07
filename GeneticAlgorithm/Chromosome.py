# from https://github.com/exowanderer/vaelstmpredictor/blob/GeneticAlgorithm/Genetic-Algorithm.py
# python vaelstmpredictor/genetic_algorithm_vae_predictor.py ga_vae_nn_test_0 --verbose --num_generations 500 --population_size 10 --num_epochs 200
import numpy as np
import os
# import random

from contextlib import redirect_stdout

from keras import backend as K
from keras.utils import to_categorical

from sklearn.externals import joblib
from time import time

from vaelstmpredictor.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmpredictor.utils.model_utils import save_model_in_pieces
from vaelstmpredictor.utils.model_utils import AnnealLossWeight
from vaelstmpredictor.utils.data_utils import MNISTData
from vaelstmpredictor.utils.weightnorm import data_based_init
from vaelstmpredictor.vae_predictor.dense_model import VAEPredictor
from vaelstmpredictor.vae_predictor.train import train_vae_predictor

def debug_message(message): print('[DEBUG] {}'.format(message))
def info_message(message): print('[INFO] {}'.format(message))

class Chromosome(VAEPredictor):
    # params = ["size_vae_hidden1", "size_vae_hidden2", "size_vae_hidden3", 
    #             "size_vae_hidden4", "size_vae_hidden5", 
    #           "vae_latent_dim", 
    #           "size_dnn_hidden1", "size_dnn_hidden2", "size_dnn_hidden3", 
    #             "size_dnn_hidden4", "size_dnn_hidden5"]

    def __init__(self, clargs, data_instance, vae_latent_dim, 
                vae_hidden_dims, dnn_hidden_dims, 
                generationID = 0, chromosomeID = 0, 
                vae_kl_weight = 1.0, vae_weight = 1.0, 
                dnn_weight = 1.0, dnn_kl_weight = 1.0, 
                verbose = False):

        self.verbose = verbose
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
        
        self.build_model()
        # self.model.compile(optimizer=self.optimizer)
        self.neural_net = self.model
        self.fitness = 0
        self.isTrained = False
        
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
                    do_ckpt = self.clargs.do_ckpt)

        if self.clargs.kl_anneal > 0: 
            self.vae_kl_weight = K.variable(value=0.1)
        if self.clargs.w_kl_anneal > 0: 
            self.dnn_kl_weight = K.variable(value=0.0)
        
        # self.clargs.optimizer, was_adam_wn = init_adam_wn(self.clargs.optimizer)
        # self.clargs.optimizer = 'adam' if was_adam_wn else self.clargs.optimizer
        
        save_model_in_pieces(self.model, self.clargs)
        
        vae_train = DI.data_train
        vae_features_val = DI.data_valid

        data_based_init(self.model, DI.data_train[:self.clargs.batch_size])

        vae_labels_val = [DI.labels_valid, predictor_validation, 
                            predictor_validation,DI.labels_valid]

        validation_data = (vae_features_val, vae_labels_val)
        train_labels = [DI.labels_train, predictor_train, predictor_train, DI.labels_train]
        
        print('\n\nFITTING MODEL\n\n')

        self.history = self.model.fit(vae_train, train_labels,
                                    shuffle = True,
                                    epochs = self.clargs.num_epochs,
                                    batch_size = self.clargs.batch_size,
                                    callbacks = callbacks,
                                    validation_data = validation_data)

        max_kl_anneal = max(self.clargs.kl_anneal, self.clargs.w_kl_anneal)
        self.best_ind = np.argmin([x if i >= max_kl_anneal + 1 else np.inf \
                    for i,x in enumerate(self.history.history['val_loss'])])
        
        self.best_loss = {k: self.history.history[k][self.best_ind] \
                                        for k in self.history.history}
        
        # self.best_val_loss = sum([val for key,val in self.best_loss.items() \
        #                             if 'val_' in key and 'loss' in key])
        
        self.fitness = 1.0 / self.best_loss['val_loss']
        self.isTrained = True
        
        if verbose: 
            print('\n\n')
            print("Generation: {}".format(self.generationID))
            print("Chromosome: {}".format(self.chromosomeID))
            print("Operation Time: {}".format(time() - start_train))
            print('\nBest Loss:')
            for key,val in self.best_loss.items():
                print('{}: {}'.format(key,val))

            print('\nFitness: {}'.format(self.fitness))
            print('\n\n')
        
        joblib_save_loc ='{}/{}_{}_{}_trained_model_output_{}.joblib.save'
        self.joblib_save_loc = joblib_save_loc.format(self.model_dir, 
                                        self.run_name, self.generationID, 
                                        self.chromosomeID, self.time_stamp)

        wghts_save_loc = '{}/{}_{}_{}_trained_model_weights_{}.save'
        self.wghts_save_loc = wghts_save_loc.format(self.model_dir, 
                                        self.run_name, self.generationID, 
                                        self.chromosomeID, self.time_stamp)
        
        model_save_loc = '{}/{}_{}_{}_trained_model_full_{}.save'
        self.model_save_loc = model_save_loc.format(self.model_dir, 
                                        self.run_name, self.generationID, 
                                        self.chromosomeID, self.time_stamp)
        
        self.neural_net.save_weights(self.wghts_save_loc, overwrite=True)
        self.neural_net.save(self.model_save_loc, overwrite=True)

        try:
            joblib.dump({'best_loss':self.best_loss,
                            'history':self.history}, 
                            self.joblib_save_loc)
        except Exception as e:
            print(str(e))
