from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Random Password'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://Username:Password@Username.mysql.pythonanywhere-services.com/databasename'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "x-access-token"], supports_credentials=True)
db = SQLAlchemy(app)

class Chromosome(db.Model):
    __tablename__ = 'Chromosome'
    id = db.Column(db.Integer, primary_key=True)
    chromosomeID = db.Column(db.Integer, default = 0)
    generationID = db.Column(db.Integer, default = 0)
    isTrained = db.Column(db.Integer, default = 0)
    fitness = db.Column(db.Float, default = -1)
    run_name = db.Column(db.String(50), default = 'dummy')
    predictor_type = db.Column(db.Integer, default = 'classification')
    batch_size = db.Column(db.Integer, default = 128)
    optimizer = db.Column(db.String(50), default = 'adam')
    num_epochs = db.Column(db.Integer, default = 200)
    dnn_weight = db.Column(db.Float, default = 1.0)
    vae_weight = db.Column(db.Float, default = 1.0)
    vae_kl_weight = db.Column(db.Float, default = 1.0)
    dnn_kl_weight = db.Column(db.Float, default = 1.0)
    prediction_log_var_prior = db.Column(db.Float, default = 0.0)
    patience = db.Column(db.Integer, default = 10)
    kl_anneal = db.Column(db.Integer, default = 0)
    w_kl_anneal = db.Column(db.Integer, default = 0)
    dnn_log_var_prior = db.Column(db.Float, default = 0.0)
    log_dir = db.Column(db.String(50), default = 'data/logs')
    model_dir = db.Column(db.String(50), default = 'data/model')
    table_dir = db.Column(db.String(50), default = 'data/tables')
    train_file = db.Column(db.String(50), default = 'MNIST')
    cross_prob = db.Column(db.Float, default = 0.7)
    mutate_prob = db.Column(db.Float, default = 0.01)
    population_size = db.Column(db.Integer, default = 200)
    num_generations = db.Column(db.Integer, default = 100)
    time_stamp = db.Column(db.Integer, default = 0)
    hostname = db.Column(db.String(50), default = '127.0.0.1')
    val_vae_reconstruction_loss = db.Column(db.Float, default = 1)
    val_vae_latent_args_loss = db.Column(db.Float, default = 1)
    val_dnn_latent_layer_loss = db.Column(db.Float, default = 1)
    val_dnn_latent_mod_loss = db.Column(db.Float, default = 1)
    num_vae_layers = db.Column(db.Integer, default = 0)
    num_dnn_layers = db.Column(db.Integer, default = 0)
    size_vae_latent = db.Column(db.Integer, default = 0)
    size_vae_hidden = db.Column(db.Integer, default = 0)
    size_dnn_hidden = db.Column(db.Integer, default = 0)
    num_conv_layers = db.Column(db.Integer, default = 0)
    size_kernel = db.Column(db.String(50), default = '[]')
    size_pool = db.Column(db.String(50), default = '[]')
    size_filter = db.Column(db.String(50), default = '[]')
    info = db.Column(db.String(100), default = '')

class Variables(db.Model):
    __tablename__ = 'Variables'
    name = db.Column(db.String(50), primary_key=True)
    value = db.Column(db.Integer, default = 0)
