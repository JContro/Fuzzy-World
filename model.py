import pandas as pd
import utils as u
from model_utils import Model

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
###################################
# Ingesting the dataset
###################################

SPREADSHEET_PATH = '~/Desktop/Data/Matching_process/fuzzy_match_app_check.xlsx'

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
# Train Random Forest model
###################################

my_model = Model(X_train, y_train, X_dev, y_dev)
my_model.train_RF()

# print(my_model.grid_search())
# The best model has the following parameters:
# {'n_estimators': 1600, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}