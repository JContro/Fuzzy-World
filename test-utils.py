import utils as u
import pandas as pd

SPREADSHEET_PATH = '~/Desktop/Data/Matching_process/advertisers_sponsors.xlsx'
advertisers = pd.read_excel(SPREADSHEET_PATH, sheet_name=0)

words_to_ignore = ['capital', 'management', 'partners', 'group', '&', 'asset', 'investment', 'of',
                       'fund', 'services', 'insurance', 'global', 'financial', 'the', 'investments', 'consulting',
                       'bank', 'international', 'solutions', 'wealth', 'pension', 'associates', 'media', 'new',
                       'london', 'risk',
                       'securities', 'real', 'gaming', 'estate', 'trust', 'co', 'office', 'family', 'company', 'de',
                       'research', 'funds', 'foundation', ]

common_words = []
with open("1-1000.txt") as f:
    for line in f:
        common_words.append(line.rstrip())


advertisers['Acronym'] = advertisers['Advertiser'].apply(lambda x: u.get_acronym(x, words_to_ignore, common_words))
print(advertisers)