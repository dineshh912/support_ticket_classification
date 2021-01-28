from flask import render_template, request, jsonify
from app.main import bp
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import re
import string
import joblib
import warnings

warnings.filterwarnings('ignore')

# Model File Location #######################

cnn_own_embedding = os.path.abspath("app/static/models/1D_CNN_model_with_training_own_embedding.h5")
cnn_glov_embedding = os.path.abspath("app/static/models/CNN_model_with_pre_trained_embedding_v3.h5")
cnn_own_embedding_dropout = os.path.abspath("app/static/models/CNN_model_with_pre_trained_embedding_dropout_v1.h5")
cnn_glov_embedding_dropout = os.path.abspath("app/static/models/CNN_model_with_training_own_embedding_dropout_v1.h5")
doc2vec = os.path.abspath("app/static/models/logistic_doc2vec_v3.pkl")
logistic_regression = os.path.abspath("app/static/models/logistic_regression_v3.pkl")
multinomial = os.path.abspath("app/static/models/multinomial_naive_bayes_v3.pkl")
random = os.path.abspath("app/static/models/random_forest_v3.pkl")

label_map = {0: 'Account service',
                 1: 'Credit card or prepaid card',
                 2: 'Credit reporting',
                 3: 'Debt collection',
                 4: 'Loans',
                 5: 'Money transfer, VC and Others',
                 6: 'Mortgage'}

key_to_label_name = [x[1] for x in sorted(label_map.items())]

# Clean Text ################################

def clean_text(doc):
    """
      1. Converting all text into lower case
      2. Removing classified words like xxx
      3. Remove stop words
      4. remove punctuation
      5. remove digits
      6. Wordnet lemmatizer
      7. Custom regex for further cleaning
      """
    # Set stop word as english
    stop_word = set(stopwords.words('english'))
    # Tokenize the sentence and make all character lower case
    doc = [x.lower() for x in word_tokenize(doc)]
    # Remove classified texts
    doc = [x for x in doc if x.lower() != 'xxxx' and x.lower() != 'xx'
           and x.lower() != 'xx/xx/xxxx' and x.lower() != 'xxxx/xxxx/xxxx']
    # Remove stop words
    doc = [x for x in doc if x not in stop_word]
    # Remove Punctuation
    doc = [x for x in doc if x not in string.punctuation]
    # Remove Digits
    doc = [x for x in doc if not x.isdigit()]
    # Set NLTK Wordnet lemmatizer and lemmatize the sentence
    lemmatizer = WordNetLemmatizer()
    doc = " ".join([lemmatizer.lemmatize(word) for word in doc])

    # Regular expression to remove unwanted chars
    pattern_1 = re.compile(r"(\W)|(\d)")
    pattern_2 = re.compile(r"\s\s+")

    doc = pattern_1.sub(" ", doc)
    doc = pattern_2.sub("", doc)
    return doc

# Index ######################################

@bp.route('/')
def index():
    return render_template('index.html')

# Sample #####################################


@bp.route('/sample')
def sample():
    return render_template('sample.html')


# Prediction ##################################


@bp.route('/classify', methods=["POST"])
def classify():
    text = request.form['text_narrative']
    processed_text = clean_text(text)
    model = joblib.load(multinomial)
    result = model.predict([processed_text])
    results = {"class": key_to_label_name[result.item()]}

    return jsonify(results)
