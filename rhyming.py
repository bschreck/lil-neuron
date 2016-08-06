import pronouncing as pron
ALL_PHONES = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                  "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                  "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                  "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
CONSONANT_BEGINS = set(["B", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "V", "W", "Y", "Z"])
CONSONANTS = set([c for c in ALL_PHONES if c[0] in CONSONANT_BEGINS])
VOWELS = [p for p in ALL_PHONES if p not in CONSONANTS]

def alliteration(word1, word2):
    prons1, max_w = get_pronunciations(word1, include_stresses=False,
                                phone_encoding_func=None)
    prons2, max_w = get_pronunciations(word2, include_stresses=False,
                                phone_encoding_func=None)
    starting_vowels1 = [vowel(phones[0]) for phones in prons1 if vowel(phones[0])]
    starting_vowels2 = [vowel(phones[0]) for phones in prons2 if vowel(phones[0])]
    print starting_vowels1
    print starting_vowels2

def vowel(phone):
    if phone in VOWELS:
        return phone
    else:
        return False


def get_pronunciations(word, include_stresses=False,
                       phone_encoding_func=None):
    max_phones_per_word = 0
    if not phone_encoding_func:
        phone_encoding_func = lambda x:x
    pronunciations = []
    for pr in pron.phones_for_word(word):
        expanded = []
        for phone_stress in pr.split(' '):
            phone, stress = split_phone_stress(phone_stress)
            if include_stresses:
                expanded.append([phone_encoding_func(phone), stress])
            else:
                expanded.append(phone_encoding_func(phone))
        len_phones = len(expanded)
        if len_phones > max_phones_per_word:
            max_phones_per_word = len_phones
        pronunciations.append(expanded)
    return pronunciations, max_phones_per_word

def split_phone_stress(phone_stress):
    try:
        stress = int(phone_stress[-1])
    except:
        return phone_stress, 0
    else:
        return phone_stress[:-1], stress


