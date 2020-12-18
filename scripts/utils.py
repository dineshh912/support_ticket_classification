import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def tokenize_sentence(sentence: str, stop_words=True, punctuation=True, numbers=True, classified=True)->list:
    """
    # https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/

    Tokenize a given string, and return the words as a list.
    The function offers functionality to exclude the words that are either
    1) a stopword 2) punctuation symbol 3) a number or 4) has the format 'XX'
    or 'XXXX' indicates the words that classififed
    """
    stop_word = set(stopwords.words('english')) 

    tokenized = [x.lower() for x in word_tokenize(sentence)]
    
    if classified:
        tokenized = [x for x in tokenized if x.lower() != 'xxxx' and
                    x.lower() != 'xx' and x.lower() != 'xx/xx/xxxx']
    
    if stop_words:
        tokenized = [x for x in tokenized if x not in stop_word]
     
    if punctuation:
        tokenized = [x for x in tokenized if x not in string.punctuation]
    
    if numbers:
        tokenized = [x for x in tokenized if not x.isdigit()]
        
    return tokenized



def lemmatize_sentence(sentence, return_form = 'string'):
    """
    Lemmatize a given string .
    # https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
	# https://programmerbackpack.com/lemmatization-and-stemming-in-nlp-the-complete-practical-guide/
    
    Input:
    ------
        sentence: 
            Sentence that we want to lemmatize each word. The input can be
            of the form of tokens (list) or the complete sentence (string).
        return_form: 
            Format of the return function. Can be either a string
            with the concatenated lemmatized words or a list of the 
            lemmatized words.
    Returns:
    -------
        If join_string = True then the function returns the
        lemmatized words as a sentence. Else it returns the words as a list.
    """
    # Handle the case where the input is the string without being tokenized
    if type(sentence) != list:
        sentence = re.findall(r"[\w']+|[.,!?;]", sentence)

    lemmatizer = WordNetLemmatizer()
    if return_form == 'string':
        return ' '.join([lemmatizer.lemmatize(word) for word in sentence])
    else:
        return [lemmatizer.lemmatize(word) for word in sentence]


def save_as_pickle(file_name, path_to_join, data):
	"""
	Save the processed data into pickle file
	Input:
	------
		file_name:
			file name which needs to be assign on the pickle file
		path_to_join:
			file path where pickle file needs to be stored.
		data:
			Dataframe

	Returns:
	-------
		return str "failed/success"
	"""
	try:
		pickled_file_loc = os.path.join(path_to_join, file_name)
		data.to_pickle(pickled_file_loc)

	except Exception as e:
		return str(e)


def plotConfusionMatrixHeatmap(input_df: pd.core.frame.DataFrame, model_name: str, figsize=(20, 18)):
    """
    Return the results of a multiclass classification algorithms as a heatmap
    based on a confusion matrix.        
    """
    # Heatmap of the results
    plt.figure(figsize=figsize)
    sns.heatmap(input_df, annot=True, fmt='d', cmap='Reds')
    plt.ylabel('True', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', size=14, fontweight='bold')
    plt.show()