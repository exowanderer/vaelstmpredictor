
# A very simple Flask Hello World app for you to get started with...

from flask import request, jsonify
from database import Chromosome, Variables, db, app

@app.route('/')
def index():
    return 'Works!'

@app.route('/AddChrom')
def AddChrom():
    if request.method == 'GET':
        chromosomeID = request.args.get('chromosomeID')
        generationID = request.args.get('generationID')
        isTrained = request.args.get('isTrained')
        fitness = request.args.get('fitness')
        run_name = request.args.get('run_name')
        predictor_type = request.args.get('predictor_type')
        batch_size = request.args.get('batch_size')
        optimizer = request.args.get('optimizer')
        num_epochs = request.args.get('num_epochs')
        dnn_weight = request.args.get('dnn_weight')
        vae_weight = request.args.get('vae_weight')
        vae_kl_weight = request.args.get('vae_kl_weight')
        dnn_kl_weight = request.args.get('dnn_kl_weight')
        prediction_log_var_prior = request.args.get('prediction_log_var_prior')
        patience = request.args.get('patience')
        kl_anneal = request.args.get('kl_anneal')
        w_kl_anneal = request.args.get('w_kl_anneal')
        dnn_log_var_prior = request.args.get('dnn_log_var_prior')
        log_dir = request.args.get('log_dir')
        model_dir = request.args.get('model_dir')
        table_dir = request.args.get('table_dir')
        train_file = request.args.get('train_file')
        cross_prob = request.args.get('cross_prob')
        mutate_prob = request.args.get('mutate_prob')
        population_size = request.args.get('population_size')
        num_generations = request.args.get('num_generations')
        time_stamp = request.args.get('time_stamp')
        hostname = request.args.get('hostname')
        num_vae_layers = request.args.get('num_vae_layers')
        num_dnn_layers = request.args.get('num_dnn_layers')
        size_vae_latent = request.args.get('size_vae_latent')
        size_vae_hidden = request.args.get('size_vae_hidden')
        size_dnn_hidden = request.args.get('size_dnn_hidden')

        c = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chromosomeID, Chromosome.generationID == generationID).first()
        if(c == None):
            chrom = Chromosome(chromosomeID = chromosomeID,
                                generationID = generationID,
                                isTrained = isTrained,
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
                                time_stamp = time_stamp,
                                hostname = hostname,
                                num_vae_layers = num_vae_layers,
                                num_dnn_layers = num_dnn_layers,
                                size_vae_latent = size_vae_latent,
                                size_vae_hidden = size_vae_hidden,
                                size_dnn_hidden = size_dnn_hidden)
            db.session.add(chrom)
        else :
            c.isTrained = isTrained
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
            c.time_stamp = time_stamp
            c.hostname = hostname
            c.num_vae_layers = num_vae_layers
            c.num_dnn_layers = num_dnn_layers
            c.size_vae_latent = size_vae_latent
            c.size_vae_hidden = size_vae_hidden
            c.size_dnn_hidden = size_dnn_hidden
        db.session.commit()
        return "1"
    return '0'

@app.route('/GetGeneration')
def GetGeneration():
    if request.method == 'GET':
        generationID = db.session.query(Variables).filter(Variables.name == "CurrentGen").first().value
        chroms = db.session.query(Chromosome).filter(Chromosome.generationID == generationID).all()
        resp = []
        for c in chroms:
            resp.append({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
                        'isTrained': c.isTrained,
            			'fitness': c.fitness,
            			'run_name': c.run_name,
            			'predictor_type': c.predictor_type,
            			'batch_size': c.batch_size,
            			'optimizer': c.optimizer,
            			'num_epochs': c.num_epochs,
            			'dnn_weight': c.dnn_weight,
            			'vae_weight': c.vae_weight,
            			'vae_kl_weight': c.vae_kl_weight,
            			'dnn_kl_weight': c.dnn_kl_weight,
            			'prediction_log_var_prior': c.prediction_log_var_prior,
            			'patience': c.patience,
            			'kl_anneal': c.kl_anneal,
            			'w_kl_anneal': c.w_kl_anneal,
            			'dnn_log_var_prior': c.dnn_log_var_prior,
            			'log_dir': c.log_dir,
            			'model_dir': c.model_dir,
            			'table_dir': c.table_dir,
            			'train_file': c.train_file,
            			'cross_prob': c.cross_prob,
            			'mutate_prob': c.mutate_prob,
            			'population_size': c.population_size,
            			'num_generations': c.num_generations,
            			'time_stamp': c.time_stamp,
            			'hostname': c.hostname,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden})
        return jsonify(resp)
    return '0'

@app.route('/GetUnTrainedChrom')
def GetUnTrainedChrom():
    if request.method == 'GET':
        c = db.session.query(Chromosome).filter(Chromosome.isTrained == 0).first()
        if(c == None):
            c = db.session.query(Chromosome).filter(Chromosome.isTrained == 1).first()
        if(c == None):
            return "0"
        c.isTrained = 1
        db.session.commit()
        return jsonify({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
                        'isTrained': c.isTrained,
            			'fitness': c.fitness,
            			'run_name': c.run_name,
            			'predictor_type': c.predictor_type,
            			'batch_size': c.batch_size,
            			'optimizer': c.optimizer,
            			'num_epochs': c.num_epochs,
            			'dnn_weight': c.dnn_weight,
            			'vae_weight': c.vae_weight,
            			'vae_kl_weight': c.vae_kl_weight,
            			'dnn_kl_weight': c.dnn_kl_weight,
            			'prediction_log_var_prior': c.prediction_log_var_prior,
            			'patience': c.patience,
            			'kl_anneal': c.kl_anneal,
            			'w_kl_anneal': c.w_kl_anneal,
            			'dnn_log_var_prior': c.dnn_log_var_prior,
            			'log_dir': c.log_dir,
            			'model_dir': c.model_dir,
            			'table_dir': c.table_dir,
            			'train_file': c.train_file,
            			'cross_prob': c.cross_prob,
            			'mutate_prob': c.mutate_prob,
            			'population_size': c.population_size,
            			'num_generations': c.num_generations,
            			'time_stamp': c.time_stamp,
            			'hostname': c.hostname,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden})
    return '0'

@app.route('/GetIsDone')
def GetIsDone():
    if request.method == 'GET':
        isDone = db.session.query(Variables).filter(Variables.name == "isDone").first()
        if isDone == None:
            return "0"
        return isDone.value
