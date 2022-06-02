from flask_pydantic import validate
from flask import request
from flask import Flask
from flask import make_response
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import pickle

# Le dataframe contenant les données sur les strokes qu'on a mis dans un pickle
app = Flask(__name__)
auth = HTTPBasicAuth()
input_resultat = open('output.pickle', "rb")
input_perf = open('perf.pickle', "rb")

df = pickle.load(input_resultat)
perf = pickle.load(input_perf)

# Users exemple

users = {
         'alice': generate_password_hash("wonderland"),
         'bob': generate_password_hash("builder"),
         'clementine': generate_password_hash("mandarine")
        }

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username

@app.route("/")
@auth.login_required
def index():
    return "<h2> Bienvenue, %s! Ceci est l'API pour le projet : Strokes! </h2>" % auth.current_user()
@app.route("/status")
def status():
    return make_response("<h2> L'api est bien fonctionnel! </h2>",200)

@app.route("/modele", methods=["POST", "GET"])
@validate()
def modele():
    res = df.to_json(orient='index')
    return '''
           Vous êtes bien connecté! <br>
           Voici le resultat après application de la regression logistique sur strokes.csv : <br>
           {}
           '''.format(res)

@app.route("/performance", methods=["POST", "GET"])
@validate()
def performance():
  return '''
           Vous êtes bien connecté! <br>
           Voici la matrice de confusion après application de la regression logistique  : <br>
           {}
           '''.format(perf)

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
