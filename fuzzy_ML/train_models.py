import pandas as pd
import numpy as np
import dataset_utils as u
import model_utils as m
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import pickle

# Import training dataset

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
with open("../1-1000.txt") as f:
    for line in f:
        common_words.append(line.rstrip())


###################################
# Create a new dataset with the features to learn from
###################################

base_df = pd.DataFrame()

base_df['name_a'] = df['name_clean_co_pam'].str.lower()
base_df['name_b'] = df['match_option'].str.lower()

# we are training the model so we need to add the target column
base_df['accept_match'] = df['accept_match']

base_df = u.prepare_dataset(base_df, training=True)


###################################
# Split the dataset into train, dev, test set
###################################

base_df = base_df.sample(frac=1).reset_index(drop=True)

# Separate majority and minority classes
majority_df = base_df[base_df.target == base_df["target"].value_counts().index[0]]
minority_df = base_df[base_df.target == base_df["target"].value_counts().index[-1]]

# Downsample majority class
majority_downsampled_df = resample(
    majority_df,
    replace=False,  # sample without replacement
    n_samples=len(minority_df),  # to match minority class
    random_state=123,
)  # reproducible results



# Combine minority class with downsampled majority class
downsampled_df = pd.concat([majority_downsampled_df, minority_df])

# Display new class counts
downsampled_df.target.value_counts()


non_numerical_cols = ["name_a", "name_b", "acr_a", "acr_b"]
feature_columns = [
    "acr_match",
    "JW_distance",
    "LV_distance",
    "num_words_a",
    "num_words_b",
    "len_a",
    "len_b",
]
target_column = "target"

X = downsampled_df.drop("target", axis=1)
y = downsampled_df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=43
)

###################################
# Train a random forest classifier
###################################

# best_params = rfc.grid_search()
# print(best_params)
# I have trained the RF model and these are the best parameters
params = {'n_estimators': 1200,
          'min_samples_split': 2,
          'min_samples_leaf': 2,
          'max_features': 'sqrt',
          'max_depth': 20,
          'bootstrap': True}

print("Training a Random Forest Classifier")
rfc = m.Random_forest(X_train[feature_columns], y_train, X_test[feature_columns], y_test, params)
rfc.evaluate_model(plot=True)
rfc_trained = rfc.get_trained_model()
m.save_model(rfc_trained, "./trained_models/rfc.sav")



###################################
# XGBoost
###################################

print("Training Catboost")
catb = m.Catboost(X_train[feature_columns], y_train, X_test[feature_columns], y_test)
catb.evaluate_model(plot=True)
catb_trained = catb.get_trained_model()
m.save_model(catb_trained, "./trained_models/catb.sav")


