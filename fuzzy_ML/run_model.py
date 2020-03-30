# Imports

import pandas as pd
import numpy as np
import model_utils as m
import dataset_utils as u

pd.set_option("display.max_columns", 101)

# Data imports
investors_path = "~/Desktop/Data/Matching_process/investors_to_merge.csv"
investors_df = pd.read_csv(investors_path)

managers_path = "~/Desktop/Data/Matching_process/managers_to_merge.csv"
managers_df = pd.read_csv(managers_path)

target_table_path = "~/Desktop/Data/Matching_process/table_to_merge_onto.csv"
target_df = pd.read_csv(target_table_path)

# Name columns
investors_col = "name_clean"
managers_col = "name_clean"
target_col = "name_clean_co"
#
# print(investors_df.head())
# print(target_df.head())
#
# print(target_df.company_type.unique())
# print(len(target_df))
# print(len(target_df[target_df['company_type'].isna()]))

#############################
# Load the pretrained model
#############################

catb = m.load_model("./trained_models/catb.sav")

#############################
# Matching the investors
#############################

investor_target_df = target_df[target_df['company_type'] == "investor"]

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


c = 0
first = True
for df in u.generate_matching_df(investors_df, investors_col, investor_target_df, target_col):
    feature_df = u.prepare_dataset_pred(df)
    feature_df['Prediction'] = catb.predict(feature_df[feature_columns])
    matches_df = df[df['Prediction'] == 1]
    if first:
        first = False
        final_match_df = matches_df
    else:
        final_match_df.append(matches_df, ignore_index=True)


    c+=1

    if c%3 == 0:
        break


print(final_match_df.reset_index())