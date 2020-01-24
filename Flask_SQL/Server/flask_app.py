
# A very simple Flask Hello World app for you to get started with...

from flask import request, jsonify, render_template
from database import Chromosome, Variables, db, app
from sqlalchemy import func
import numpy as np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/AddChrom')
def AddChrom():
    if request.method == 'GET':

        c = db.session.query(Chromosome).filter(Chromosome.chromosomeID == request.args.get('chromosomeID'),
                                                Chromosome.generationID == request.args.get('generationID'),
                                                Chromosome.num_vae_layers == request.args.get('num_vae_layers'),
                                                Chromosome.num_dnn_layers == request.args.get('num_dnn_layers'),
                                                Chromosome.size_vae_latent == request.args.get('size_vae_latent'),
                                                Chromosome.size_vae_hidden == request.args.get('size_vae_hidden'),
                                                Chromosome.size_dnn_hidden == request.args.get('size_dnn_hidden'),
                                                Chromosome.size_kernel == request.args.get('size_kernel'),
                                                Chromosome.size_pool == request.args.get('size_pool'),
                                                Chromosome.size_filter == request.args.get('size_filter')).first()
        if(c == None):
            return '1'

        for key in request.args.keys():
            setattr(c, key, request.args.get(key))

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
            			'info': c.info,
                        'l1_coef': c.l1_coef,
                        'l2_coef': c.l2_coef,
                        'dropout_rate': c.dropout_rate})
            trained += (c.isTrained == 2)
            taken += (c.isTrained == 1)
            not_taken += (c.isTrained == 0)
            generation = (c.generationID +1)

        dic = {"5-Chroms": resp, "3-Taken": taken, "2-Trained": trained, "4-Not Taken": not_taken, "1-Generation": generation}
        return jsonify(dic)
    return '0'

def obj2dict(obj, default=0):
    ret = {}
    for col in obj.__table__.columns:
        if(getattr(obj, col.name) == None):
            ret[col.name] = default
        else:
            ret[col.name] = getattr(obj, col.name)
    return ret

@app.route('/GetDatabase')
def GetDatabase():
    if request.method == 'GET':
        chroms = db.session.query(Chromosome).all()
        resp = []
        for c in chroms:
            resp.append(obj2dict(c))
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
        return jsonify(obj2dict(c))
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
            fitness = c.fitness
            if(fitness < min_val):
                fitness = min_val + np.random.uniform(-1, 1)

            fitness = ((1/fitness) - (1/max_val))/((1/min_val) - (1/max_val))
            fitness = np.sqrt(np.sqrt(fitness))
            fitness = 0 if np.isnan(fitness) else fitness
            nodes.append({"generation": c.generationID,
                            "name": c.chromosomeID,
                            "fitness": fitness,
                            "size": c.num_conv_layers,
                            "height": c.num_dnn_layers})
        links = []
        for c in chroms:
            if c.fitness > 0:
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