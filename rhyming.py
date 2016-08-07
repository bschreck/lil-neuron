import pronouncing
import itertools
import numpy as np
ALL_PHONES = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                  "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                  "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                  "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
CONSONANT_BEGINS = set(["B", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "V", "W", "Y", "Z"])
CONSONANTS = set([c for c in ALL_PHONES if c[0] in CONSONANT_BEGINS])
VOWELS = set([p for p in ALL_PHONES if p not in CONSONANTS])
LEVEL1_EQUIV_VOWEL_SETS = [
        set(["AA", "AH"]),
        ]
LEVEL2_EQUIV_VOWEL_SETS = [
        set(["AA", "AO", "AH", "AW", "OW", "OY", "UH", "UW"]),
        set(["AE", "AH", "AY", "EH", "ER", "EY", "UH"]),
    ]
LEVEL1_EQUIV_CONSONANT_SETS = [
    set(['B', 'M', 'P', 'V', 'F']),
    set(['CH', 'JH', 'SH', 'ZH', 'S']),
    set(['D', 'DH', 'T', 'TH']),
    set(['G', 'K']),
    set(['HH']),
    set(['L']),
    set(['N', 'NG']),
    set(['R']),
    set(['W', 'V']),
    set(['Y'])
    ]
def make_equiv_sounds(sound_sets, vowels=True):
    full_set = VOWELS
    if not vowels:
        full_set = CONSONANTS
    equiv = {}
    for phone in full_set:
        equiv[phone] = set([phone])
        for equiv_set in sound_sets:
            if phone in equiv_set:
                equiv[phone] |= equiv_set
    return equiv
EQUIV_CONSONANTS = make_equiv_sounds(LEVEL1_EQUIV_CONSONANT_SETS, vowels=False)
EQUIV_VOWELS1 = make_equiv_sounds(LEVEL1_EQUIV_VOWEL_SETS, vowels=True)
EQUIV_VOWELS2 = make_equiv_sounds(LEVEL2_EQUIV_VOWEL_SETS, vowels=True)

# want 1st and 2nd level close consonants
# Come up with scoring system for how closely a group of phonemes rhymes

def match_prons(prons):
    """
    Most pronunciations will only be different in one or two vowels
    So make a new structure holding the different phones for each index in the word
    """
    # vowels in 99% of cases (maybe 100%) can't be doubled up, so
    # match on ordering of vowels

    # structure is [[cons_group1, cons_group2], [vowel1, vowel2], ...]
    matched = []
    vc_iterator = vowel_consonant_iterator(prons[0])
    for group, vowel in vc_iterator:
        matched.append(([group], vowel))
    starts_with_vowel = matched[0][1]
    for p in prons[1:]:
        vc_iterator = vowel_consonant_iterator(p)
        offset = 0
        for i, gv in enumerate(vc_iterator):
            group, vowel = gv
            if i == 0 and vowel != starts_with_vowel:
                if vowel:
                    offset = 1
                else:
                    matched.insert(0, ([group], vowel))
                    continue
            if i < len(matched):
                existing_groupings = matched[i + offset][0]
                found_match = False
                for g in existing_groupings:
                    if all((x == group[j] for j, x in enumerate(g))):
                        found_match = True
                        break
                if not found_match:
                    matched[i + offset][0].append(group)
            else:
                matched.append(([group], vowel))
    return matched

def vowel_consonant_iterator(phones):
    for k, phone_group in itertools.groupby(phones, is_vowel):
        yield list(phone_group), k

def multisyllabic_rhyme_score(word1, word2):
    prons1, max_w = get_pronunciations(word1, include_stresses=False,
                                phone_encoding_func=None)
    prons2, max_w = get_pronunciations(word2, include_stresses=False,
                                phone_encoding_func=None)
    prons1 = match_prons(prons1)
    prons2 = match_prons(prons2)
    print prons1
    print prons2
    closest_match_score = 0
    for start1, gv1 in enumerate(prons1):
        for start2, gv2 in enumerate(prons2):
            if start1 == start2 and start1 != 0:
                # if words start on same index they "align"
                # shorter subsequences of "aligned" subwords
                # can't possibly have a greater score than longer ones
                # so only need to check the longest fone for the aligned case
                continue
            cur_score = multi_subscore(prons1[start1:], prons2[start2:])
            if cur_score > closest_match_score:
                closest_match_score = cur_score
    return closest_match_score

def multi_subscore(phones1, phones2):
    score = 0
    # [([['N', 'F']], False), ([['AH'], ['AA']], True)...]
    # [([['N', 'F']], False), ([['AO']], True)...]

    # for each phoneme group (either group of consonants or a vowel)
    print "="*50
    print phones1
    print phones2
    for i, phone_group1_v in enumerate(phones1):
        phone_group1 = phone_group1_v[0]
        if i >= len(phones2):
            break
        phone_group2 = phones2[i][0]
        # for each pronunciation
        max_pron_subscore = 0
        for pron1 in phone_group1:
            for pron2 in phone_group2:
                # if its a consonant group there may be
                # a different number of consonants
                # determine score based on highest subscore
                # of either lining up from the start or lining
                # up from the end, ignoring possibility that
                # best score could be from the middle for
                # computational reasons
                if len(pron1) > len(pron2):
                    diff = len(pron1) - len(pron2)
                    matched11 = pron1[diff:]
                    matched12 = pron2

                    matched21 = pron1[:len(pron2)]
                    matched22 = pron2
                else:
                    diff = len(pron2) - len(pron1)
                    matched11 = pron1
                    matched12 = pron2[diff:]

                    matched21 = pron1
                    matched22 = pron2[:len(pron1)]
                subscore1 = 0
                subscore2 = 0
                for j, m in enumerate(matched11):
                    subscore1 += phone_similarity(m, matched12[j])
                    subscore2 += phone_similarity(matched21[j], matched22[j])
                pron_score = max(subscore1, subscore2)
                if pron_score > max_pron_subscore:
                    # print matched11, matched12, subscore1
                    # print matched21, matched22, subscore2
                    max_pron_subscore = pron_score
                #print max_pron_subscore
        score += max_pron_subscore
        print score
    print "="*50
    return score

def phone_similarity(phone1, phone2):
    phone1_is_vowel = is_vowel(phone1)
    phone2_is_vowel = is_vowel(phone2)
    # consonant and vowel
    if phone1_is_vowel != phone2_is_vowel:
        return 0
    # equal
    elif phone1 == phone2 and phone1_is_vowel:
        return 5
    elif phone1 == phone2:
        return 2
    elif phone1_is_vowel:
        # level1 vowel similarity
        if phone2 in EQUIV_VOWELS1[phone1] or phone1 in EQUIV_VOWELS1[phone2]:
            return 3
        # level2 vowel similarity
        elif phone2 in EQUIV_VOWELS2[phone1] or phone2 in EQUIV_VOWELS2[phone2]:
            return 2
        else:
            # just vowels
            return 1
    else:
        if phone2 in EQUIV_CONSONANTS[phone1] or phone1 in EQUIV_CONSONANTS[phone2]:
            #level1 consonant similarity
            return 1
        else:
            # just consonant
            return 1

def is_vowel(phone):
    if phone in VOWELS:
        return True
    else:
        return False


def get_pronunciations(word, include_stresses=False,
                       phone_encoding_func=None):
    max_phones_per_word = 0
    if not phone_encoding_func:
        phone_encoding_func = lambda x:x
    pronunciations = []
    for pr in pronouncing.phones_for_word(word):
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


