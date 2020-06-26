import os
from sys import getsizeof

from flask import Flask, request
from flask_cors import CORS, cross_origin
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from google.cloud import storage
from google.oauth2 import service_account
import torch
from nmt_model import *
from run import beam_search
import nltk
import tempfile

# f = tempfile.NamedTemporaryFile(mode='w')
# f.write(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
# f.flush()
# storage_client = storage.Client.from_service_account_json(f.name)
# f.close()
# bucket = storage_client.get_bucket('glove-vectors-300d')
# blob = bucket.blob('glove.6B.300d.txt')

# with open('static/glove.twitter.27B.50d.txt', 'wb') as glove_temp:
#     storage_client.download_blob_to_file(blob, glove_temp)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# glove_file = 'api/static/glove/glove.6B.300d.txt'
glove_file = 'static/glove.twitter.27B.50d.txt'
# glove_file = 'static/glove_example.txt'
# glove.twitter.27B.50d.txt = get_tmpfile("glove.6B.300d.word2vec.txt")
# glove.twitter.27B.50d.txt = get_tmpfile("glove.twitter.27B.50d.txt")
model = None
with open("glove.twitter.27B.50d.txt", 'wb') as word2vec_glove_file:
# glove.twitter.27B.50d.txt = get_tmpfile("glove_example.txt")
    glove2word2vec(glove_file, "glove.twitter.27B.50d.txt")
    print(getsizeof(word2vec_glove_file))
    model = KeyedVectors.load_word2vec_format("glove.twitter.27B.50d.txt")
    print(getsizeof(model))

params = torch.load('static/model.bin', map_location=lambda storage, loc: storage)
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
@cross_origin()
def send_ASL():
    print(request.json['englishInputText'])
    line = request.json['englishInputText']

    outputText = ''

    try:
        outputText = ' '.join(
            beam_search(nmt_model, [nltk.word_tokenize(line)], beam_size=5,
                        max_decoding_time_step=70)[0][0].value)
    except Exception as e:
        print(e)
        outputText = 'Could not translate...'
    return {'outputText': outputText}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 80))
    # app.run()
#     , static_folder='./analogies', static_url_path='/'
