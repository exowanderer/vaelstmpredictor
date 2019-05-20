from flask import request, jsonify
from database import Chromosome, db, app

@app.route('/')
def index():
    return 'Works!'

@app.route('/AddChrom')
def AddChrom():
    if request.method == 'GET':
        chromosomeID = request.args.get('chromosomeID')
        generationID = request.args.get('generationID')
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
        # do_log = request.args.get('do_log')
        do_chckpt = request.args.get('do_chckpt')
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
        iterations = request.args.get('iterations')
        # verbose = request.args.get('verbose')
        # make_plots = request.args.get('make_plots')
        time_stamp = request.args.get('time_stamp')
        hostname = request.args.get('hostname')
        num_vae_layers = request.args.get('num_vae_layers')
        num_dnn_layers = request.args.get('num_dnn_layers')
        size_vae_latent = request.args.get('size_vae_latent')
        size_vae_hidden = request.args.get('size_vae_hidden')
        size_dnn_hidden = request.args.get('size_dnn_hidden')
        isTrained = request.args.get('isTrained')

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
                            # do_log = do_log,
                            do_chckpt = do_chckpt,
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
                            iterations = iterations,
                            # verbose = verbose,
                            # make_plots = make_plots,
                            time_stamp = time_stamp,
                            hostname = hostname,
                            num_vae_layers = num_vae_layers,
                            num_dnn_layers = num_dnn_layers,
                            size_vae_latent = size_vae_latent,
                            size_vae_hidden = size_vae_hidden,
                            size_dnn_hidden = size_dnn_hidden,
                            isTrained = isTrained)
        db.session.add(chrom)
        db.session.commit()
        return "1"
    return '0'

@app.route('/GetDatabase')
def GetDatabase():
    if request.method == 'GET':
        chroms = db.session.query(Chromosome).all()

        # if(c == None): return '0'

        resp = []
        for c in chroms:
            resp.append({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
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
            # 			'do_log': c.do_log,
            			'do_chckpt': c.do_chckpt,
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
            			'iterations': c.iterations,
            # 			'verbose': c.verbose,
            # 			'make_plots': c.make_plots,
            			'time_stamp': c.time_stamp,
            			'hostname': c.hostname,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'isTrained':c.isTrained})
        return jsonify(resp)
    return '0'

@app.route('/GetChromosome')
def GetChromosome():
    if request.method == 'GET':
        chromosomeID = request.args.get('chromosomeID')
        generationID = request.args.get('generationID')

        c = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chromosomeID, Chromosome.generationID == generationID).first()

        if(c == None): return '0'

        return jsonify({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
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
            # 			'do_log': c.do_log,
            			'do_chckpt': c.do_chckpt,
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
            			'iterations': c.iterations,
            # 			'verbose': c.verbose,
            # 			'make_plots': c.make_plots,
            			'time_stamp': c.time_stamp,
            			'hostname': c.hostname,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'isTrained': c.isTrained})
    return '0'

@app.route('/GetGeneration')
def GetGeneration():
    if request.method == 'GET':
        generationID = request.args.get('generationID')

        chroms = db.session.query(Chromosome).filter(Chromosome.generationID == generationID).all()

        # if(c == None): return '0'

        resp = []
        for c in chroms:
            resp.append({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
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
            # 			'do_log': c.do_log,
            			'do_chckpt': c.do_chckpt,
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
            			'iterations': c.iterations,
            # 			'verbose': c.verbose,
            # 			'make_plots': c.make_plots,
            			'time_stamp': c.time_stamp,
            			'hostname': c.hostname,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'isTrained':c.isTrained})
        return jsonify(resp)
    return '0'

@app.route('/Reset')
def Reset():
    db.engine.execute("Drop table Chromosome")
    db.create_all()
    return "1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
