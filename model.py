import pandas as pd
import numpy as np
import utils as u
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
###################################
# Ingesting the dataset
###################################

#SPREADSHEET_PATH = '~/Desktop/Data/Matching_process/fuzzy_match_app_check.xlsx'
SPREADSHEET_PATH = '/Users/jameshawkins/Desktop/project/fuzzymatch/Fuzzy-World/fuzzy_match_app_check.xlsx'

df = pd.read_excel(SPREADSHEET_PATH, sheet_name=0)

###################################
# Preparing list for NLP
###################################

# These are words to
words_to_ignore = ['capital', 'management', 'partners', 'group', '&', 'asset', 'investment', 'of', \
                   'fund', 'services', 'insurance', 'global', 'financial', 'the', 'investments', 'consulting', 'bank', \
                   'international', 'solutions', 'wealth', 'pension', 'associates', 'media', 'new', 'london', 'risk', \
                   'securities', 'real', 'gaming', 'estate', 'trust', 'co', 'office', 'family', 'company', 'de', \
                   'research', 'funds', 'foundation']

# These common words will be used in searching for acronyms
common_words = []
with open("1-1000.txt") as f:
    for line in f:
        common_words.append(line.rstrip())

###################################
# Create a new dataset with the features to learn from
###################################

base_df = pd.DataFrame()

# Retaining the two target columns
base_df['name_a'] = df['name_clean_co_pam']
base_df['name_b'] = df['match_option']

# Get the acronyms for a and b
base_df['acr_a'] = base_df['name_a'].apply(lambda x: u.get_acronym(x, words_to_ignore, common_words))
base_df['acr_b'] = base_df['name_b'].apply(lambda x: u.get_acronym(x, words_to_ignore, common_words))

# Create a numerical field for the same acronyms
base_df['acr_match'] = base_df.apply(lambda row: u.acronym_checker(row['acr_a'], row['acr_b']), axis=1)

# Get the number of words
base_df['num_words_a'] = base_df['name_a'].apply(lambda x: u.get_number_words(x))
base_df['num_words_b'] = base_df['name_b'].apply(lambda x: u.get_number_words(x))

# Get the length of the strings
base_df['len_a'] = base_df['name_a'].apply(lambda x: len(str(x)))
base_df['len_b'] = base_df['name_b'].apply(lambda x: len(str(x)))

# Get Jaro Winkler distance
base_df['JW_distance'] = base_df.apply(lambda row: u.jaro_winkler_distance(row['name_a'], row['name_b']), axis=1)

# Get Levenshtein distance
base_df['LV_distance'] = base_df.apply(lambda row: u.levenshtein_distance(row['name_a'], row['name_b']), axis=1)

# Get the target
base_df['target'] = df['accept_match'].apply(lambda x: u.convert_target(x))

###################################
# Split the dataset into train, dev, test set
###################################

base_df = base_df.sample(frac=1).reset_index(drop=True)

non_numerical_cols = ['name_a', 'name_b', 'acr_a', 'acr_b']
feature_columns = ['acr_match', 'JW_distance', 'LV_distance', 'num_words_a', 'num_words_b',
                   'len_a', 'len_b']
target_column = 'target'

train_df = base_df.iloc[:2600, :]
X_train = train_df[feature_columns]
y_train = train_df[target_column]

dev_df = base_df.iloc[2600:3000, :]
X_dev = dev_df[feature_columns]
y_dev = dev_df[target_column]

test_df = base_df.iloc[3000:, :]
X_test = test_df[feature_columns]
y_test = test_df[target_column]

###################################
# Bench Mark Fuzzy Matching
###################################

df_dict = []

def fuzzy(df,level=50):
    '''
    Fuzzy match name_a and name_b columns at variable ratio levels to 
    create a baseline reference to accuracy
    '''
    c = 0
    for a_index, a_row in df.iterrows():
        
        a_conf = str(a_row['name_a']).lower().split()
        
        if len(a_conf) > 1:
            a_match = ' '.join([t for t in a_conf if t not in words_to_ignore])
     
        else:
            a_match = a_conf[0]
            
        b_conf = str(a_row['name_b']).lower().split()
            
        if len(b_conf) > 1:
            b_match = ' '.join([t for t in b_conf if t not in words_to_ignore])
     
        else:
            b_match = b_conf[0]    
                    
        ratio = fuzz.ratio(str(a_match),str(b_match))
        
        if ratio > level:
            assignment = 1
        else:
            assignment = 0
        
        # creating dictionary to track matches          
        dict_ = {'name_a':a_row[0],'name_b':a_row[1],\
                 'similarity':ratio,'ratio_level':level,'target': a_row[11],\
                 'assignment': assignment}
       
        df_dict.append(dict_)            
            
        c += 1
        if c % 100 == 0:
            print(c)
    return df_dict
        


# you need to refresh the df_dict EVERY ITERATION OF fuzzy            
df_dict = []            
# train set
train_50 = pd.DataFrame(fuzzy(train_df,50))
train_60 = pd.DataFrame(fuzzy(train_df,60))
train_70 = pd.DataFrame(fuzzy(train_df,70))
train_80 = pd.DataFrame(fuzzy(train_df,80))
train_90 = pd.DataFrame(fuzzy(train_df,90))
train_100 = pd.DataFrame(fuzzy(train_df,100))

# you need to refresh the df_dict EVERY ITERATION OF fuzzy            
df_dict = []       
# dev set
dev_50 = pd.DataFrame(fuzzy(dev_df,50))
dev_60 = pd.DataFrame(fuzzy(dev_df,60))
dev_70 = pd.DataFrame(fuzzy(dev_df,70))
dev_80 = pd.DataFrame(fuzzy(dev_df,80))
dev_90 = pd.DataFrame(fuzzy(dev_df,90))
dev_100 = pd.DataFrame(fuzzy(dev_df,100))

# you need to refresh the df_dict EVERY ITERATION OF fuzzy            
df_dict = []       
# test sets            
test_50 = pd.DataFrame(fuzzy(test_df,50))
test_60 = pd.DataFrame(fuzzy(test_df,60))
test_70 = pd.DataFrame(fuzzy(test_df,70))
test_80 = pd.DataFrame(fuzzy(test_df,80))
test_90 = pd.DataFrame(fuzzy(test_df,90))
test_100 = pd.DataFrame(fuzzy(test_df,100))

# Iterating to create DF for plotting
test_lst = [test_50, test_60, test_70,test_80,test_90]
dev_lst = [dev_50, dev_60, dev_70,dev_80,dev_90]
train_lst = [train_50, train_60, train_70,train_80,train_90]

# Confusion Matri for each ratio level 


def con_matri(df):
    return confusion_matrix(df['target'],df['assignment'])

# accuracy: (tp + tn) / (p + n)
accuracy_test = []

accuracy_test.append(accuracy_score(test_50['target'],test_50['assignment']))
accuracy_test.append(accuracy_score(test_60['target'],test_60['assignment']))
accuracy_test.append(accuracy_score(test_70['target'],test_70['assignment']))
accuracy_test.append(accuracy_score(test_80['target'],test_80['assignment']))
accuracy_test.append(accuracy_score(test_90['target'],test_90['assignment']))
accuracy_test.append(accuracy_score(test_100['target'],test_100['assignment']))

accuracy_dev = []

accuracy_dev.append(accuracy_score(dev_50['target'],dev_50['assignment']))
accuracy_dev.append(accuracy_score(dev_60['target'],dev_60['assignment']))
accuracy_dev.append(accuracy_score(dev_70['target'],dev_70['assignment']))
accuracy_dev.append(accuracy_score(dev_80['target'],dev_80['assignment']))
accuracy_dev.append(accuracy_score(dev_90['target'],dev_90['assignment']))
accuracy_dev.append(accuracy_score(dev_100['target'],dev_100['assignment']))

accuracy_train = []

accuracy_train.append(accuracy_score(train_50['target'],train_50['assignment']))
accuracy_train.append(accuracy_score(train_60['target'],train_60['assignment']))
accuracy_train.append(accuracy_score(train_70['target'],train_70['assignment']))
accuracy_train.append(accuracy_score(train_80['target'],train_80['assignment']))
accuracy_train.append(accuracy_score(train_90['target'],train_90['assignment']))
accuracy_train.append(accuracy_score(train_100['target'],train_100['assignment']))

# recall: tp / (tp + fn)
recall_test = []

recall_test.append(recall_score(test_50['target'],test_50['assignment']))
recall_test.append(recall_score(test_60['target'],test_60['assignment']))
recall_test.append(recall_score(test_70['target'],test_70['assignment']))
recall_test.append(recall_score(test_80['target'],test_80['assignment']))
recall_test.append(recall_score(test_90['target'],test_90['assignment']))
recall_test.append(recall_score(test_100['target'],test_100['assignment']))

recall_dev = []

recall_dev.append(recall_score(dev_50['target'],dev_50['assignment']))
recall_dev.append(recall_score(dev_60['target'],dev_60['assignment']))
recall_dev.append(recall_score(dev_70['target'],dev_70['assignment']))
recall_dev.append(recall_score(dev_80['target'],dev_80['assignment']))
recall_dev.append(recall_score(dev_90['target'],dev_90['assignment']))
recall_dev.append(recall_score(dev_100['target'],dev_100['assignment']))

recall_train = []

recall_train.append(recall_score(train_50['target'],train_50['assignment']))
recall_train.append(recall_score(train_60['target'],train_60['assignment']))
recall_train.append(recall_score(train_70['target'],train_70['assignment']))
recall_train.append(recall_score(train_80['target'],train_80['assignment']))
recall_train.append(recall_score(train_90['target'],train_90['assignment']))
recall_train.append(recall_score(train_100['target'],train_100['assignment']))

# precision tp / (tp + fp)
precision_test = []

precision_test.append(precision_score(test_50['target'],test_50['assignment']))
precision_test.append(precision_score(test_60['target'],test_60['assignment']))
precision_test.append(precision_score(test_70['target'],test_70['assignment']))
precision_test.append(precision_score(test_80['target'],test_80['assignment']))
precision_test.append(precision_score(test_90['target'],test_90['assignment']))
precision_test.append(precision_score(test_100['target'],test_100['assignment']))

precision_dev = []

precision_dev.append(precision_score(dev_50['target'],dev_50['assignment']))
precision_dev.append(precision_score(dev_60['target'],dev_60['assignment']))
precision_dev.append(precision_score(dev_70['target'],dev_70['assignment']))
precision_dev.append(precision_score(dev_80['target'],dev_80['assignment']))
precision_dev.append(precision_score(dev_90['target'],dev_90['assignment']))
precision_dev.append(precision_score(dev_100['target'],dev_100['assignment']))

precision_train = []

precision_train.append(precision_score(train_50['target'],train_50['assignment']))
precision_train.append(precision_score(train_60['target'],train_60['assignment']))
precision_train.append(precision_score(train_70['target'],train_70['assignment']))
precision_train.append(precision_score(train_80['target'],train_80['assignment']))
precision_train.append(precision_score(train_90['target'],train_90['assignment']))
precision_train.append(precision_score(train_100['target'],train_100['assignment']))

# =============================================================================
# Why can't the below work then be replicated? 'assignment' issue - dont understand
# =============================================================================
## When running there will be KeyError: 'target' due to empty 100 dataset i.e. no 100% matches
#test = []
#c = 50
#for i in test_lst:    
#    dict_ = {'Set': c, 'Accuracy':precision_score(i['target'],i['assignment'])}
#    test.append(dict_)
#    final_test = pd.DataFrame(test)
#    c += 10
#    

# Plotting Accuracies 
plt.plot(range(50,110,10), accuracy_test,color = 'g', label = 'Test Set')
plt.plot(range(50,110,10), accuracy_dev, color = 'y', label = 'Dev Set')
plt.plot(range(50,110,10),accuracy_train, color = 'b', label = 'Training Set' )
plt.xlabel('Fuzzy Match Limit')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Fuzzy Matching across different ratio limits - Accuracy')

plt.show()

# Plotting Recall 
plt.plot(range(50,110,10), recall_test,color = 'g', label = 'Test Set')
plt.plot(range(50,110,10), recall_dev, color = 'y', label = 'Dev Set')
plt.plot(range(50,110,10),recall_train, color = 'b', label = 'Training Set' )
plt.xlabel('Fuzzy Match Limit')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Fuzzy Matching across different ratio limits - Recall')

plt.show()

# Plotting Precision 
plt.plot(range(50,110,10), precision_test,color = 'g', label = 'Test Set')
plt.plot(range(50,110,10), precision_dev, color = 'y', label = 'Dev Set')
plt.plot(range(50,110,10),precision_train, color = 'b', label = 'Training Set' )
plt.xlabel('Fuzzy Match Limit')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Fuzzy Matching across different ratio limits - Precision')

plt.show()





###################################
# Train Random Forest model
###################################

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

train_rf_predictions = rfc.predict(X_train)
train_rf_probs = rfc.predict_proba(X_train)[:, 1]

# Actual class predictions
rfc_predict = rfc.predict(X_dev)
# Probabilities for each class
rfc_probs = rfc.predict_proba(X_dev)[:, 1]


# Calculate roc auc
roc_value = roc_auc_score(y_dev, rfc_probs)

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline['recall'] = recall_score(y_dev,
                                      [1 for _ in range(len(y_dev))])
    baseline['precision'] = precision_score(y_dev,
                                            [1 for _ in range(len(y_dev))])
    baseline['roc'] = 0.5

    results = {}

    results['recall'] = recall_score(y_dev, predictions)
    results['precision'] = precision_score(y_dev, predictions)
    results['roc'] = roc_auc_score(y_dev, probs)

    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_dev, [1 for _ in range(len(y_dev))])
    model_fpr, model_tpr, _ = roc_curve(y_dev, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


evaluate_model(rfc_predict, rfc_probs, train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')
