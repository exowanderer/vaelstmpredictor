from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'This is The Secret Key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://LAUDeepGenerativ:123456789VaeLstmPredictor123456789@LAUDeepGenerativeGenetics.mysql.pythonanywhere-services.com/LAUDeepGenerativ$default'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization",
                    "Access-Control-Allow-Credentials",
                    "Access-Control-Allow-Origin",
                    "Access-Control-Allow-Headers",
                    "x-access-token"],
     supports_credentials=True)

db = SQLAlchemy(app)

class Chromosome(db.Model):
    __tablename__ = 'Chromosome'
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.Integer, default=0)
    date_taken = db.Column(db.Integer, default=0)
    date_trained = db.Column(db.Integer, default=0)
    fitness = db.Column(db.Float, default=-1)
    val_fitness = db.Column(db.Float, default=-1)
    train_reconstruction_loss = db.Column(db.Float, default=-1)
    train_kl_loss = db.Column(db.Float, default=-1)
    test_reconstruction_loss = db.Column(db.Float, default=-1)
    test_kl_loss = db.Column(db.Float, default=-1)
    info = db.Column(db.String(100), default='')

    #------------------------- Training Info --------------------------
    cross_prob = db.Column(db.Float, default=0.7)
    mutate_prob = db.Column(db.Float, default=0.01)
    population_size = db.Column(db.Integer, default=200)
    batch_size = db.Column(db.Integer, default=128)
    num_epochs = db.Column(db.Integer, default=100)
    chroms_per_loop = db.Column(db.Integer, default=10)
    #-------------------------------------------------------------------

    #----------------------------- GENES -------------------------------
    #Encoder
    num_cnn_encoder = db.Column(db.Integer, default=0)
    size_kernel_encoder = db.Column(db.String(150), default='[]')
    size_pool_encoder = db.Column(db.String(150), default='[]')
    size_filter_encoder = db.Column(db.String(150), default='[]')
    batchnorm_encoder = db.Column(db.String(150), default='[]')

    num_dnn_encoder = db.Column(db.Integer, default=0)
    size_dnn_encoder = db.Column(db.String(150), default='[]')
    l1_dnn_encoder = db.Column(db.String(150), default='[]')

    #Decoder
    # num_cnn_decoder = db.Column(db.Integer, default=0)
    # size_kernel_decoder = db.Column(db.String(150), default='[]')
    # size_pool_decoder = db.Column(db.String(150), default='[]')
    # size_filter_decoder = db.Column(db.String(150), default='[]')
    # num_dnn_decoder = db.Column(db.Integer, default=0)
    # size_dnn_decoder = db.Column(db.String(150), default='[]')

    #Latent
    size_latent = db.Column(db.Integer, default=0)
    size_resnet = db.Column(db.Integer, default=0)
    #-------------------------------------------------------------------


class Variables(db.Model):
    __tablename__ = 'Variables'
    name = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Integer, default=0)
