import pandas as pd
import numpy as np
import jellyfish


# These are words to ignore
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



def get_acronym(name, words_to_ignore, common_words):
    """
    Extracts the first 4 letter acronym from a company name
    :param name: name of a company
    :param words_to_ignore: list of words to ignore (common words in companies)
    :param common_words: list of common english words
    :return: acronym
    """

    words = str(name).split()

    for word in words:
        word = word.lower()

        if len(word) < 5 and word not in words_to_ignore and word not in common_words:
            return word

    return None


def get_number_words(name):
    """
    Get the number of words in a string
    :param name: name of a company
    :return: number of individual words
    """
    words = str(name).split()
    num_words = len(words)

    return num_words


def convert_target(target):
    if target == 'y':
        return 1
    elif target == 'n':
        return 0
    else:
        print(f"The target {target} is not in y or n")
        assert 1 == 0


def jaro_winkler_distance(str_a, str_b):
    """
    Returns the jaro-winkler distance of the two strings
    :param str_a: first string
    :param str_b: second string
    :return: jaro-winkle distance
    """

    return jellyfish.jaro_winkler(str(str_a), str(str_b))


def levenshtein_distance(str_a, str_b):
    """
    Returns the Levenshtein distance
    :param str_a: first string
    :param str_b: second string
    :return: Levenshtein distance
    """

    return jellyfish.levenshtein_distance(str(str_a), str(str_b))


def acronym_checker(acr_a, acr_b):
    """

    :param acr_a:
    :param acr_b:
    :return:
    """

    if acr_a is not None and acr_b is not None:
        if acr_a == acr_b:
            return 1
        else:
            return 0
    else:
        return 0

def prepare_dataset(base_df, training=False):
    """
    Uses the functions in these utils to get the features for the model to train on
    :param df:
    :return:
    """
    # Get the acronyms for a and b
    base_df['acr_a'] = base_df['name_a'].apply(lambda x: get_acronym(x, words_to_ignore, common_words))
    base_df['acr_b'] = base_df['name_b'].apply(lambda x: get_acronym(x, words_to_ignore, common_words))

    # Create a numerical field for the same acronyms
    base_df['acr_match'] = base_df.apply(lambda row: acronym_checker(row['acr_a'], row['acr_b']), axis=1)

    # Get the number of words
    base_df['num_words_a'] = base_df['name_a'].apply(lambda x: get_number_words(x))
    base_df['num_words_b'] = base_df['name_b'].apply(lambda x: get_number_words(x))

    # Get the length of the strings
    base_df['len_a'] = base_df['name_a'].apply(lambda x: len(str(x)))
    base_df['len_b'] = base_df['name_b'].apply(lambda x: len(str(x)))

    # Get Jaro Winkler distance
    base_df['JW_distance'] = base_df.apply(lambda row: jaro_winkler_distance(row['name_a'], row['name_b']), axis=1)

    # Get Levenshtein distance
    base_df['LV_distance'] = base_df.apply(lambda row: levenshtein_distance(row['name_a'], row['name_b']), axis=1)

    if training:
        # Get the target
        base_df['target'] = base_df['accept_match'].apply(lambda x: convert_target(x))
    return base_df


def get_matches(name_a, column_b):
    """

    :param name_a:
    :param column_b:
    :return:
    """

    df = pd.DataFrame({'name_a': name_a.lower(), 'name_b': column_b.str.lower()})

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


def prepare_dataset_pred(base_df):
    """

    :param df:
    :return:
    """
    # Get the acronyms for a and b
    base_df['acr_a'] = base_df['name_a'].apply(lambda x: get_acronym(x, words_to_ignore, common_words))
    base_df['acr_b'] = base_df['name_b'].apply(lambda x: get_acronym(x, words_to_ignore, common_words))

    # Create a numerical field for the same acronyms
    base_df['acr_match'] = base_df.apply(lambda row: acronym_checker(row['acr_a'], row['acr_b']), axis=1)

    # Get the number of words
    base_df['num_words_a'] = base_df['name_a'].apply(lambda x: get_number_words(x))
    base_df['num_words_b'] = base_df['name_b'].apply(lambda x: get_number_words(x))

    # Get the length of the strings
    base_df['len_a'] = base_df['name_a'].apply(lambda x: len(str(x)))
    base_df['len_b'] = base_df['name_b'].apply(lambda x: len(str(x)))

    # Get Jaro Winkler distance
    base_df['JW_distance'] = base_df.apply(lambda row: jaro_winkler_distance(row['name_a'], row['name_b']), axis=1)

    # Get Levenshtein distance
    base_df['LV_distance'] = base_df.apply(lambda row: levenshtein_distance(row['name_a'], row['name_b']), axis=1)

    return base_df