from pymongo import MongoClient
import pronouncing
import pdb
all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                  "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                  "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                  "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
def split_phone_stress(phone_stress):
    try:
        stress = int(phone_stress[-1])
    except:
        return phone_stress, 0
    else:
        return phone_stress[:-1], stress

def valid_pron(pron):
    for i, p in enumerate(pron):
        phone, stress = split_phone_stress(p)
        if stress > 2:
            print "Stress too high on {}th phone".format(i)
            return False
        elif phone not in all_phones:
            print "Unrecognized {}th phone: {}".format(i, phone)
            return False
    return True

def find_pronunciations():
    session_count = 0
    for word, count in SLANG_WORD_COUNTS:
        inp = None
        pron = None
        while inp not in ['c', 'm']:
            print "SLANG WORD:"
            print "==> {} (count = {})".format(word, count)
            print "Combine words (c) or manual (m):"

            inp = raw_input("==> ")
            if inp == 'c':
                print "Enter words:"
                words = raw_input("==> ").split()
                combined_prons = []
                for w in words:
                    while True:
                        existing_slang = db.slang_words.find({'word':w}).limit(1)
                        existing_slang = [s['pronunciation'] for s in existing_slang if 'pronunciation' in s]
                        if len(existing_slang):
                            prons = [' '.join(existing_slang[0])]
                        elif w in PARTIAL_WORDS:
                            prons = PARTIAL_WORDS[w]
                        else:
                            prons = pronouncing.phones_for_word(w)
                        print "{} prons:".format(w)
                        print prons
                        winp = raw_input("Select pron (int), new_word, start over (s), manual (m): ")
                        if not winp:
                            winp = 0
                        elif winp == 's':
                            # start over with new words
                            inp = None
                            break
                        elif winp == 'm':
                            while True:
                                pron = raw_input("Enter pron: ").split()
                                valid = valid_pron(pron)
                                if valid:
                                    break
                            string_pron = ' '.join(pron)
                            if w not in PARTIAL_WORDS:
                                PARTIAL_WORDS[w] = [string_pron]
                                db.partial_words.insert_one({'word': w, 'pronunciation': [string_pron]})
                            elif string_pron not in PARTIAL_WORDS[w]:
                                PARTIAL_WORDS[w].append(string_pron)
                                db.partial_words.update({'word': w}, {'$set': {'pronunciation': PARTIAL_WORDS[w]}})
                            combined_prons.extend(pron)
                            break
                        try:
                            pron_index = int(winp)
                        except:
                            w = winp
                        else:
                            wpron = prons[pron_index]
                            combined_prons.extend(wpron.split())
                            break
                    if not inp:
                        break
                pron = combined_prons
            elif inp == 'm':
                while True:
                    pron = raw_input("Enter pron: ").split()
                    valid = valid_pron(pron)
                    if valid:
                        break
            else:
                print "Unknown option"
            if pron:
                print pron
                print "updating"
                session_count += 1
                print "SESSION COUNT: {}".format(session_count)
                db.slang_words.update({'word': word}, {'$set':{'pronunciation':pron}})


def load_dicts():
    global client
    global db
    client = MongoClient()
    db = client['lil-neuron-db']

    records = db.slang_words.find({"pronunciation": {'$exists': False}})
    global SLANG_WORD_COUNTS
    SLANG_WORD_COUNTS = sorted([(r["word"], r["count"]) for r in records],
                               key=lambda x: -x[1])

    # records = db.partial_words.find()
    global PARTIAL_WORDS
    # PARTIAL_WORDS = {r['word']: r['pronunciation'] for r in records}
    PARTIAL_WORDS = {}

    records = db.dwords.find()
    global WORD_COUNTS
    WORD_COUNTS = sorted([(r["word"], r["count"]) for r in records],
                         key=lambda x: -x[1])


    global WORD_TO_DWORD_INT
    WORD_TO_DWORD_INT = {}
    records = db.word_to_dwordint.find()
    for r in records:
        WORD_TO_DWORD_INT[r["word"]] = r["int_list"]

    global DWORD_TO_INT
    global INT_TO_DWORD
    DWORD_TO_INT = {}
    INT_TO_DWORD = {}
    records = db.dword_to_int.find()
    for r in records:
        INT_TO_DWORD[r["int"]] = r["word"]
if __name__ == '__main__':
    load_dicts()
    find_pronunciations()
