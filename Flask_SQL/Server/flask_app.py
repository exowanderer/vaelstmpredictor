
# A very simple Flask Hello World app for you to get started with...

from flask import request, jsonify, render_template
from database import Chromosome, Variables, db, app
from sqlalchemy import func

@app.route('/')
def index():
    return render_template('index.html')

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
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        run_time = request.args.get('run_time')
        hostname = request.args.get('hostname')
        val_vae_reconstruction_loss = request.args.get('val_vae_reconstruction_loss')
        val_vae_latent_args_loss = request.args.get('val_vae_latent_args_loss')
        val_dnn_latent_layer_loss = request.args.get('val_dnn_latent_layer_loss')
        val_dnn_predictor_layer_loss = request.args.get('val_dnn_predictor_layer_loss')
        num_vae_layers = request.args.get('num_vae_layers')
        num_dnn_layers = request.args.get('num_dnn_layers')
        size_vae_latent = request.args.get('size_vae_latent')
        size_vae_hidden = request.args.get('size_vae_hidden')
        size_dnn_hidden = request.args.get('size_dnn_hidden')
        num_conv_layers = request.args.get('num_conv_layers')
        size_kernel = request.args.get('size_kernel')
        size_pool = request.args.get('size_pool')
        size_filter = request.args.get('size_filter')
        info = request.args.get('info')

        c = db.session.query(Chromosome).filter(Chromosome.chromosomeID == chromosomeID,
                                                Chromosome.generationID == generationID,
                                                Chromosome.num_vae_layers == num_vae_layers,
                                                Chromosome.num_dnn_layers == num_dnn_layers,
                                                Chromosome.size_vae_latent == size_vae_latent,
                                                Chromosome.size_vae_hidden == size_vae_hidden,
                                                Chromosome.size_dnn_hidden == size_dnn_hidden,
                                                Chromosome.size_kernel == size_kernel,
                                                Chromosome.size_pool == size_pool,
                                                Chromosome.size_filter == size_filter).first()
        if(c == None):
            return '1'

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
        return "1"
    return '0'

@app.route('/GetGeneration')
def GetGeneration():
    if request.method == 'GET':
        generationID = db.session.query(Variables).filter(Variables.name == "CurrentGen").first().value
        chroms = db.session.query(Chromosome).filter(Chromosome.generationID == generationID).all()
        resp = []
        trained = 0
        taken = 0
        not_taken = 0
        generation = 0
        for c in chroms:
            resp.append({'chromosomeID': c.chromosomeID,
                        'generationID': c.generationID,
                        'isTrained': c.isTrained,
            			'fitness': c.fitness,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'num_conv_layers': c.num_conv_layers,
                        'size_kernel': c.size_kernel,
                        'size_pool': c.size_pool,
                        'size_filter': c.size_filter,
            			'info': c.info})
            trained += (c.isTrained == 2)
            taken += (c.isTrained == 1)
            not_taken += (c.isTrained == 0)
            generation = (c.generationID +1)

        dic = {"5-Chroms": resp, "3-Taken": taken, "2-Trained": trained, "4-Not Taken": not_taken, "1-Generation": generation}
        return jsonify(dic)
    return '0'

@app.route('/GetDatabase')
def GetDatabase():
    if request.method == 'GET':
        chroms = db.session.query(Chromosome).all()
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
            			'start_time': c.start_time,
            			'end_time': c.end_time,
            			'run_time': c.run_time,
            			'hostname': c.hostname,
            			'val_vae_reconstruction_loss': c.val_vae_reconstruction_loss,
                        'val_vae_latent_args_loss': c.val_vae_latent_args_loss,
                        'val_dnn_latent_layer_loss': c.val_dnn_latent_layer_loss,
                        'val_dnn_predictor_layer_loss': c.val_dnn_predictor_layer_loss,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'num_conv_layers': c.num_conv_layers,
                        'size_kernel': c.size_kernel,
                        'size_pool': c.size_pool,
                        'size_filter': c.size_filter,
            			'info': c.info})
        return jsonify(resp)
    return '0'

@app.route('/GetUnTrainedChrom')
def GetUnTrainedChrom():
    if request.method == 'GET':
        c = db.session.query(Chromosome).filter(Chromosome.isTrained == 0).first()
        if(c == None):
            c = db.session.query(Chromosome).filter(Chromosome.isTrained == 1).order_by(db.func.rand()).first()
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
            			'start_time': c.start_time,
            			'end_time': c.end_time,
            			'run_time': c.run_time,
            			'hostname': c.hostname,
            			'val_vae_reconstruction_loss': c.val_vae_reconstruction_loss,
                        'val_vae_latent_args_loss': c.val_vae_latent_args_loss,
                        'val_dnn_latent_layer_loss': c.val_dnn_latent_layer_loss,
                        'val_dnn_predictor_layer_loss': c.val_dnn_predictor_layer_loss,
            			'num_vae_layers': c.num_vae_layers,
            			'num_dnn_layers': c.num_dnn_layers,
            			'size_vae_latent': c.size_vae_latent,
            			'size_vae_hidden': c.size_vae_hidden,
            			'size_dnn_hidden': c.size_dnn_hidden,
            			'num_conv_layers': c.num_conv_layers,
                        'size_kernel': c.size_kernel,
                        'size_pool': c.size_pool,
                        'size_filter': c.size_filter,
            			'info': c.info})
    return '0'

@app.route('/GetIsDone')
def GetIsDone():
    if request.method == 'GET':
        isDone = db.session.query(Variables).filter(Variables.name == "isDone").first()
        if isDone == None:
            return "0"
        return isDone.value

@app.route('/Visuals')
def Visuals():
    if request.method == 'GET':
        resp = {}
        chroms = db.session.query(Chromosome).order_by(Chromosome.generationID, Chromosome.chromosomeID).all()
        population = chroms[0].population_size

        def get_index(generationID, chromosomeID):
            return generationID*population + chromosomeID

        nodes = []
        min_val = db.session.query(func.min(Chromosome.fitness)).filter(Chromosome.fitness > 0).first()[0]
        max_val = db.session.query(func.max(Chromosome.fitness)).filter(Chromosome.fitness > 0).first()[0]
        for c in chroms:
            if(c.fitness < min_val):
                c.fitness = min_val
            c.fitness = ((1/c.fitness) - 1/max_val)/(1/min_val - 1/max_val)
            nodes.append({"generation": c.generationID,
                            "name": c.chromosomeID,
                            "fitness": c.fitness,
                            "size": c.num_conv_layers,
                            "height": c.size_filter})
        links = []
        for c in chroms:
            bold = 0 if ("Mutated" in c.info) else 1
            if("Descendant" in c.info):
                parentID = int(c.info.split(" ")[2])
                links.append({"source": get_index(c.generationID -1, parentID),
                                "target": get_index(c.generationID, c.chromosomeID),
                                "op": bold})
            elif("Child" in c.info):
                parentID1 = int(c.info.split(" ")[2])
                parentID2 = int(c.info.split(" ")[4])
                links.append({"source": get_index(c.generationID -1, parentID1),
                                "target": get_index(c.generationID, c.chromosomeID),
                                "op": bold})
                links.append({"source": get_index(c.generationID -1, parentID2),
                                "target": get_index(c.generationID, c.chromosomeID),
                                "op": bold})


        resp["nodes"] = nodes
        resp["links"] = links
        resp["population"] = population
        resp["max"] = max_val
        resp["min"] = min_val
        return jsonify(resp)
