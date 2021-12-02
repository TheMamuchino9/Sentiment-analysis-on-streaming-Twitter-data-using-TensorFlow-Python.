from tweepy import Stream, OAuthHandler
import tensorflow as tf
import json
import string
import re
import pickle
from flask import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

consumer_key = 'YOUR CREDENTIALS'
consumer_secret = 'YOUR CREDENTIALS'
access_token = 'YOUR CREDENTIALS'
access_secret = 'YOUR CREDENTIALS'
app = Flask(__name__)


class Mystreamer(Stream):
    def on_connect(self):
        self.i = 0
        global mensaje, sentimiento, estatus
        mensaje = []
        sentimiento = []
        estatus = []

    def on_data(self, raw_data):
        self.i = self.i + 1
        print(self.i)
        print("------New message-----")
        # print(raw_data)
        msg = json.loads(raw_data)
        if "extended_tweet" in msg:
            print("Extended Tweet")
            msg = msg['extended_tweet']['full_text']
        else:
            msg = msg['text']
        print(msg)
        msgx = msg
        print("\nPREPROCESSED\n")
        msg = normalization(msg)
        msg = msg.replace("\n", " ")
        msg = ['{}'.format(msg)]
        print(msg)
        # Create the sequences
        max_length = 80
        padding_type = 'post'
        msg_tokenized = tokenizer.texts_to_sequences(msg)
        msg_padded = pad_sequences(msg_tokenized, padding=padding_type, maxlen=max_length)
        print('\n PREDICTION \n')
        classes = reloaded.predict(msg_padded)
        # The closer the class is to 1, the more positive the review is deemed to be
        print(classes)
        global mensaje, sentimiento
        mensaje.append(msgx)
        sentimiento.append(float(classes[0][0]))
        if float(classes[0][0]) >= 0.5:
            estatus.append("Positive")
        else:
            estatus.append("Negative")
        print('\n')
        if self.i >= number:
            self.running = False


def normalization(text):
    hashtags = re.compile(r"^#\S+|\s#\S+")
    mentions = re.compile(r"^@\S+|\s@\S+")
    urls = re.compile(r"https?://\S+")
    rt = re.compile(r"RT")
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", re.UNICODE)
    words = urls.sub('', text)
    words = hashtags.sub('', words)
    words = mentions.sub('', words)
    words = rt.sub('', words)
    words = emoji_pattern.sub(r'', words)
    punct = set(string.punctuation)
    words = "".join([ch for ch in words if ch not in punct])
    return words.strip().lower()


def send_data(keyword):
    print('start sending data from Twitter')
    # authentication based on the credentials
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # start sending data from the Streaming API
    twitter_stream = Mystreamer(consumer_key, consumer_secret, access_token,
                                access_secret)
    twitter_stream.filter(track=keyword, languages=["en"])


@app.route('/add', methods=['POST'])
def basic():
    if request.form['submit'] == 'add':
        keyword = request.form['keyword']
        global number
        number = float(request.form['number'])
        send_data(keyword=[keyword])
    return render_template('index.html', t=mensaje, x=sentimiento, s=estatus)


@app.route('/')
def list():
    return render_template('index.html')


if __name__ == "__main__":
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    export_path_keras = "./1635913579.h5"
    reloaded = tf.keras.models.load_model(export_path_keras)
    app.run(debug=True)
    # select here the keyword for the tweet data
    # send_data(keyword=['love'])
