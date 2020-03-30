import pandas as pd
from old_shit import utils as u

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
###################################
# Ingesting the dataset
###################################

SPREADSHEET_PATH = '~/Desktop/Data/Matching_process/advertisers_sponsors.xlsx'

advertisers = pd.read_excel(SPREADSHEET_PATH, sheet_name=0)
sponsors = pd.read_excel(SPREADSHEET_PATH, sheet_name=1)
to_match = pd.read_excel(SPREADSHEET_PATH, sheet_name=2)

advertisers = advertisers.sample(frac=1)

print(advertisers.columns)
print(sponsors.columns)
print(to_match.columns)

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


def get_matches(name_a, column_b):
    """

    :param name_a:
    :param column_b:
    :return:
    """

    df = pd.DataFrame({'name_a': name_a, 'name_b': column_b})

    return df


def generate_matching_df(df_a, col_a, df_b, col_b):
    """

    :param df_a:
    :param col_a:
    :param df_b:
    :param col_b:
    :return:
    """

    for index, row in df_a.iterrows():
        df = get_matches(row[col_a], df_b[col_b])
        yield df


def prepare_dataset(base_df):
    """

    :param df:
    :return:
    """
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

    return base_df


from old_shit.test import get_trained_model

model = get_trained_model()

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
for df in generate_matching_df(advertisers, "Advertiser", to_match, "name_clean_co"):
    df = prepare_dataset(df)
    df['Prediction'] = model.predict(df[feature_columns])
    print(df[df['Prediction'] == 0])

    c += 1
    if c % 3 == 0:
        break
