import os

from flask import Flask, request, json
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from google.cloud import storage
from google.oauth2 import service_account
import torch
from api.nmt_model import *
from api.run import beam_search
import nltk
import tempfile

f = tempfile.NamedTemporaryFile(mode='w')
f.write(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
f.flush()
storage_client = storage.Client.from_service_account_json(f.name)
f.close()
bucket = storage_client.get_bucket('glove-vectors-300d')
blob = bucket.blob('glove.6B.300d.txt')

with open('api/static/glove_temp.txt', 'wb') as glove_temp:
    storage_client.download_blob_to_file(blob, glove_temp)

app = Flask(__name__, static_folder='./public', static_url_path='/')

# glove_file = 'api/static/glove/glove.6B.300d.txt'
glove_file = 'api/static/glove_temp.txt'
# word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
word2vec_glove_file = get_tmpfile("glove_temp.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

params = torch.load('api/static/model.bin', map_location=lambda storage, loc: storage)
args = params['args']
nmt_model = NMT(vocab=params['vocab'], **args)
nmt_model.load_state_dict(params['state_dict'])


def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]


@app.route('/api/analogy', methods=['POST'])
def send_analogy():
    x1, x2, y1 = request.json['x1'], request.json['x2'], request.json['y1']
    print(x1 + ' ' + x2 + ' ' + y1)
    return {'y2': (analogy(x1.lower(), x2.lower(), y1.lower())).capitalize()}
    # r = request
    # return {"y2": "Hi"}


@app.route('/api/ASL', methods=['POST'])
def send_ASL():
    print(request.json['englishInputText'])
    line = request.json['englishInputText']

    outputText = ''

    try:
        outputText = ' '.join(
            beam_search(model, [nltk.word_tokenize(line)], beam_size=5,
                        max_decoding_time_step=70)[0][0].value)
    except:
        outputText = 'Could not translate...'
    return {'outputText': outputText}


@app.route('/')
def index():
    app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 80))
    # app.run()
#     , static_folder='./analogies', static_url_path='/'
