
# A very simple Flask Hello World app for you to get started with...

from flask import request, jsonify, render_template
from database import Chromosome, db, app
from sqlalchemy import func
import numpy as np
import time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/AddChrom')
def AddChrom():
    if request.method == 'GET':

        c = db.session.query(Chromosome).filter(Chromosome.id == request.args.get('id'),
                                                Chromosome.date_created == request.args.get('date_created')).first()
        if(c == None):
            return '1'

        for key in request.args.keys():
            setattr(c, key, request.args.get(key))

        db.session.commit()
        return "1"
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
        trained = 0
        taken = 0
        not_taken = 0
        resp = []
        for c in chroms:
            resp.append(obj2dict(c))
            if(c.date_trained > 0):
                trained += 1
            elif(c.date_taken > 0):
                taken += 1
            else:
                not_taken += 1

        return jsonify({"4-Chroms": resp, "2-Taken": taken, "1-Trained": trained, "3-Not Taken": not_taken})
    return '0'

@app.route('/GetUnTrainedChrom')
def GetUnTrainedChrom():
    if request.method == 'GET':
        c = db.session.query(Chromosome).filter(Chromosome.date_taken <= 0).first()
        if(c == None):
            c = db.session.query(Chromosome).filter(Chromosome.date_trained <= 0).order_by(db.func.rand()).first()
        if(c == None):
            return "0"
        c.date_taken = int(time.time())
        db.session.commit()
        return jsonify(obj2dict(c))
    return '0'

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