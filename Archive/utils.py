import jellyfish

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
