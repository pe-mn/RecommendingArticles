# Importing necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import regex
import string
import random
import matplotlib.pyplot as plt


# Text Preprocessing Functions:
# ------------------------------

# Below are a list of auxiliary functions that remove a list of words (such as stop words) from the text, 
# apply stemming and remove words with 2 letters or less and words 21 or more letters (the longues word in the english alphabet is ‘Incomprehensibilities’, with 21 letters)
# https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python

# Function for converting into lower case
def make_lower_case(text):
    return text.lower()

# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

# def remove_non_ascii(s):
# # To remove non-ASCII characters from a string, s, use:
#     s = s.encode('ascii',errors='ignore')
# # Then convert it from bytes back to a string using:
#     s = s.decode()
    
# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

# Function for removing punctuation and any word less than 3 characters
# While keeping Currency Symbols
def remove_punctuation(text):
    ''' 
    This function keeps only letters, 
    digits, underscores, and currency symbols.
    Also it removes single characters.
    '''
    text = regex.findall(r'\w{3,}|\p{Sc}', text)
    text = " ".join(text)
    return text


def remove_short_words(text):
    text = ' '.join(word for word in text.split() if len(word)>3) # Faster
#     text = regex.sub(r'\b\w{,2}\b', '', text)
    return text


def remove_numbers(text):
    '''
    remove all text containing numbers at its beginning
    '''
    text = regex.sub(r'\d+\w*\b', '', text)
    return text

def remove_repeated_chars(text):
    '''
    remove Remove words which contains same character more than twice
    '''
    text = re.sub(r"(.)\1{2,}\w*", "", text, flags=re.I)
    return text

# ? --> optional
# If you wanted to find both http and https --> https?
# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               "❤️"
                               "🥰"
            u"\U0001F600-\U0001F64F" 
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#--------------------------------------------------

# Identify correlations between numeric features
def get_correlated_columns(df, min_corr_level=0.95):

    # correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    # triu --> triangle-upper
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than min_corr_level
    cols_to_drop = [column for column in upper.columns if any(upper[column] > min_corr_level)]

    return cols_to_drop


# To get the number of pca components to use (highest variance)
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
#     for i in range(num_components):
#         ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
    
# Interpret Principal Components
def pca_results(df, pca, component_no, n_features):    
    # Build a dataframe with features_no features capturing most variability
    # for a given component number (component_no) -->  -1 because 0-based indexing
    pca_comp = pd.DataFrame(np.round(pca.components_, 4), columns=df.keys()).iloc[component_no - 1]
    pca_comp.sort_values(ascending=False, inplace=True)
    pca_comp = pd.concat([pca_comp.head(n_features), pca_comp.tail(n_features)])
    
    # Plot the result
    pca_comp.plot(kind='bar', 
                  title='Most {} weighted features for PCA component {}'.format(n_features*2, component_no),
                  figsize=(12, 6))
    
    ax = plt.gca()
    ax.text(4, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(pca.explained_variance_ratio_[component_no-1]))
    
    plt.show()
    
    return pca_comp

# ---------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary

def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 

    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
#         print('Precision score: ', format(precision_score(y_true, preds, average='micro')))
#         print('Recall score: ', format(recall_score(y_true, preds, average='micro')))
#         print('F1 score: ', format(f1_score(y_true, preds, average='micro')))
#         print('\n\n')
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))
#         print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds, average='micro')))
#         print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds, average='micro')))
#         print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds, average='micro')))
#         print('\n\n')











