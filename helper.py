import unicodedata
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical_dtype, is_float_dtype, is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from time import strftime
import seaborn as sns

stopwords = stopwords.words('english')


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def remove_parentheses_content(text):
    text = re.sub(r'\([^)]*\)', "", text)
    return text

def replace_multiple_spaces(text):
    text = re.sub(r'[ ]{2,}'," ", text)
    return text

def remove_separated_numbers(text):
    text = re.sub(r'[.,][0-9]{1,}|([0-9]{1,}[,.][0-9]{1,})',"", text)
    return text

def remove_dates(text):
    text = re.sub(r'^(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4}|am|pm)$',"",text)
    text = re.sub(r'([\d]{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s[\d]{4})',"",text)
    return text

def remove_stopwords(text, stopword_list):
    text_list = [word for word in text if word not in stopword_list]
    return text_list

def concat_randomly(source_str, insert_str, partial = True):
    if partial != False:
        cut_pos = [pos for pos, char in enumerate(source_str) if char == " "]
        cut_pos = np.random.choice(cut_pos[1:], 1).item()
        source_str = source_str[:cut_pos]
        conc_pos = [pos for pos, char in enumerate(source_str) if char == " "]
        conc_pos = np.random.choice(conc_pos, 1).item()
        joined_str = source_str[:conc_pos] + " " + insert_str + source_str[conc_pos:]
    else:
        conc_pos = [pos for pos, char in enumerate(source_str) if char == " "]
        conc_pos = np.random.choice(conc_pos, 1).item()
        joined_str = source_str[:conc_pos] + " " + insert_str + source_str[conc_pos:]
    return joined_str


def format_dates(dates, quartiles = True, pattern = ['%d %B %Y', '%B %d %Y', '%d %b %Y', '%b %d %Y']):
    
    dates_list = []
    if quartiles != False:    
        q1 = int(np.floor(len(dates)*0.25))
        q2 = int(np.floor(len(dates)*0.5))
        q3 = int(np.floor(len(dates)*0.75))         
        for idx, val in enumerate(dates):
            if idx < q1:
                val = val.strftime(pattern[0])
                dates_list.append(val)
            elif idx > q1 and idx < q2:
                val = val.strftime(pattern[1])
                dates_list.append(val)
            elif idx > q2 and idx < q3:
                val = val.strftime(pattern[2])
                dates_list.append(val)
            elif idx > q3:
                val = val.strftime(pattern[3])
                dates_list.append(val)         
    else:
        for val in dates:
            val = val.strftime(pattern)
            dates_list.append(val)
            
    return dates_list


def tokenize(corpus, ngrams, pattern = None):
    if pattern != None:    
        if ngrams == 1:
            grams = [re.findall(pattern, doc) for doc in corpus]
            grams = [gram for sublist in grams for gram in sublist]
        elif ngrams == 2:
            grams = [b for l in corpus for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    else:
        if ngrams == 1:
            grams = [doc.split() for doc in corpus]
            grams = [gram for sublist in grams for gram in sublist]
        elif ngrams == 2:
            grams = [b for l in corpus for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    return grams


def describe_string(string):
    letters_count = 0 
    digits_count = 0
    symbols_count = 0
    length = len(string)
    istitle = string.istitle()
    for character in string:
        if character.isalpha():
            letters_count += 1
        elif character.isnumeric():
            digits_count += 1
        else:
            symbols_count += 1
    return(letters_count, digits_count, symbols_count, length, istitle)


def analyze_ngram(ngram_list):
    ngram_features = {'observation': [observation for observation in ngram_list],
                      'n_letters': [describe_string(observation)[0] for observation in ngram_list],
                      'n_digits': [describe_string(observation)[1]  for observation in ngram_list],
                      'n_symbols': [describe_string(observation)[2]  for observation in ngram_list],
                      'length': [describe_string(observation)[3] for observation in ngram_list],
                      'is_title': [describe_string(observation)[4] for observation in ngram_list],
                      'prev_word': [ngram_list[i-1] if i > 0 else np.nan for i in range(len(ngram_list))],
                      'prev_n_letters': [describe_string(ngram_list[i-1])[0] if i > 0 else np.nan for i in range(len(ngram_list))],
                      'prev_n_digits': [describe_string(ngram_list[i-1])[1]  if i > 0 else np.nan for i in range(len(ngram_list))],
                      'prev_n_symbols': [describe_string(ngram_list[i-1])[2]  if i > 0 else np.nan for i in range(len(ngram_list))],
                      'prev_length': [describe_string(ngram_list[i-1])[3] if i > 0 else np.nan for i in range(len(ngram_list))],
                      'prev_is_title': [describe_string(ngram_list[i-1])[4] if i > 0 else np.nan for i in range(len(ngram_list))],
                      'next_word': [ngram_list[i+1] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))],
                      'next_n_letters': [describe_string(ngram_list[i+1])[0] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))],
                      'next_n_digits': [describe_string(ngram_list[i+1])[1] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))],
                      'next_n_symbols': [describe_string(ngram_list[i+1])[2] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))],
                      'next_length': [describe_string(ngram_list[i+1])[3] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))],
                      'next_is_title': [describe_string(ngram_list[i+1])[4] if i < (len(ngram_list)-1) else np.nan for i in range(len(ngram_list))]
                      }
    
    ngram_features_df = pd.DataFrame(ngram_features)
    return ngram_features_df
    
# dodać opcje na wyawlnaie stopwords

def is_part_of_address(gram_df, address_df, corpus, pattern = None):
    word_idx = 0
    if pattern != None:
        docs = [re.findall(pattern, doc) for doc in corpus]
        addresses = address_df['Address'].tolist()
        address_words_lists = [re.findall(pattern, address) for address in addresses]
        for doc_idx, doc in enumerate(docs):
            address_words = address_words_lists[doc_idx]
            for doc_word in doc:
                if (doc_word in address_words) and ((gram_df.loc[word_idx, 'prev_word'] in address_words) and (gram_df.loc[word_idx, 'next_word'] in address_words)):
                    gram_df.loc[word_idx, 'is_part_of_address'] = True
                word_idx += 1
    else:
        docs = [doc.split() for doc in corpus]
        addresses = address_df['Address'].tolist()
        address_words_lists = [address.split() for address in addresses]
        for doc_idx, doc in enumerate(docs):
            address_words = address_words_lists[doc_idx]
            for doc_word in doc:
                if (doc_word in address_words) and ((gram_df.loc[word_idx, 'prev_word'] in address_words) and (gram_df.loc[word_idx, 'next_word'] in address_words)):
                    gram_df.loc[word_idx, 'is_part_of_address'] = True
                word_idx += 1
    return gram_df

    

def obj_to_cat(dataframe):
    """
    Function used to convert objects(strings) into categories
    
    Parameters:
        
    dataframe - just as the parameter name implies, expects dataframe object
    
    """
    for n, c in dataframe.items():
        if is_string_dtype(c):
            dataframe[n] = c.astype('category').cat.as_ordered()
    return dataframe


def float_to_int(dataframe):
    for n, c in dataframe.items():
        if is_float_dtype(c):
            dataframe[n] = c.astype('int')
    return dataframe


def fill_missing_nums(dataframe):    
    """
	 Function used to impute missing numerical values with column's median
	
	 Parameters:
	
	 dataframe - just as the parameter name implies, expects dataframe object
	
   	 """
    
    for n, c in dataframe.items(): 
        if is_numeric_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.median())
    return dataframe


def fill_missing_cats(dataframe):
    """
    Function used to impute missing categorical values with column's mode
	
    Parameters:
	
    dataframe - just as the parameter name implies, expects dataframe object
	
    """
    for n, c in dataframe.items():
        if is_categorical_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.mode()[0])
    return dataframe


def get_codes(dataframe):
    """
    Function for converting values of categorical variables into numbers.
    
    Parameters:
        
    dataframe - just as the parameter name implies, expects dataframe object
	
    """
    for column in dataframe.columns:
        if is_categorical_dtype(dataframe[column]):
            dataframe[column] = dataframe[column].cat.codes
    return dataframe
            
def print_score(algorithm, X_train, X_test, y_train, y_test, auc = True):
    
    # Computing and printing accuracy for training and test set
    print("Training accuracy: {}, Test accuracy: {}".format(algorithm.score(X_train, y_train), algorithm.score(X_test, y_test)))
    
    if auc == True:
        # Computing predicted probabilities: y_pred_prob
        y_pred_prob = algorithm.predict_proba(X_test)[:,1]
        # Computing and printing AUC score
        print("Area under the ROC curve: {}".format(roc_auc_score(y_test, y_pred_prob)))
        
        
def evaluate(estimator, params, X, y, scoring = 'precision'):

    # Initilize instance of estimator
    est = estimator
    
    # Set params
    est.set_params(**params)
    
    # Calc CV score
    scores = cross_val_score(estimator=est, X=X, y=y, 
                             scoring=scoring, cv=5, n_jobs = -1)
    score = np.mean(scores)
    
    return score

    

def create_confmat(algorithm, X_test, y_test, columns, colour = 'Oranges', size = (20,14)):
    
    # Computing predictions on test set
    y_pred = algorithm.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
    index = [col for col in columns], 
    columns = [col for col in columns])
    plt.figure(figsize = size)
    sns.heatmap(cm_df, annot = True, cmap = colour, fmt='g', linewidths=.2)
    plt.title('Confusion Matrix', fontsize = 20)
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    
    print("True negatives: {}  |  False negatives: {}  |  True positives: {}  |  False positives: {}".format(cm[0,0], cm[1,0], cm[1,1], cm[0,1]))  
    
    
def report(algorithm, X_test, y_test):
    
    y_pred = algorithm.predict(X_test)
    print( "Classification Report:\n{}".format(classification_report(y_test, y_pred)))    
    
def plot_auc(algorithm, X_test, y_test, size = (16,6)):

    # Computing predicted probabilities: y_pred_prob
    y_pred_prob = algorithm.predict_proba(X_test)[:,1]
    
    # Generating ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        
    # Plotting ROC curve
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = size)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.plot(fpr, tpr,)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')

    # Generating ROC curve values: precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    # Plotting PR curve
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.plot(recall, precision)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('PR Curve')
    plt.show()
    
    
def sample(df, num):
    index = sorted(np.random.permutation(len(df))[:num])
    return df.iloc[index].copy()

def predict_classes(model, X_val, y_val):
     y_pred_probs = np.argmax(model.predict_proba(X_val), axis = 1)
     y_trues = np.array(y_val)
     correct  = y_pred_probs == y_trues
     results = {'Predicted_label': y_pred_probs, 'Is_correct' : correct}
     results_df = pd.DataFrame(results)
     return results_df
 
    
def plot_feat_imp(model, dataframe, boundary = 15, best_features = False):

	"""
	Function used for plotting the most important features found by model.
	
	Parameters:
	
	model - just as the parameter name implies, expects model object
	dataframe - just as the parameter name implies, expects dataframe object
	boundary - number of features we would like to plot
	
	"""
	indices = np.argsort(model.feature_importances_)[::-1][:boundary]
	best_features_list = [col for col in dataframe.columns[indices]]

	fig = plt.figure(figsize=(9, 12))
	p = sns.barplot(y=dataframe.columns[indices][:boundary], x = model.feature_importances_[indices][:boundary], orient='h')
	p.set_xlabel("Relative importance",fontsize=12)
	p.set_ylabel("Features",fontsize=12)
	p.tick_params(labelsize=10)
	p.set_title("Feature importances")
	for i, v in enumerate(model.feature_importances_[indices][:boundary]):
		plt.text(v, i, ""+str(np.round(v,3)), color='#e59471', va='center', fontweight='bold')

	plt.show()
	
	if best_features == True:

		return best_features_list
	