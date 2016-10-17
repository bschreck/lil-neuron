from bs4 import BeautifulSoup
from bs4 import Tag
from tqdm import tqdm
import re
import sys
import numpy as np
import time
import requests
import string
from pymongo import MongoClient

"""
Scrape from lyrics.wikia.com, after first scraping the artists to use
from the top N rappers on last.fm, spotify, or echo nest, turn the artist's
name into the format lyrics.wikia.com uses, go to the artist page there, and
find the CSS class that lists all the songs (or albums) and urls Each song
also includes an Artist: at the start of each verse if the artist is not the
same as the artist who made the song, or who is a sub-artist (if it's a rap
group)
"""


def remove_punctuation(word):
    no_punc = translate_unicode(word.lower()).strip()
    return ' '.join(no_punc.split())


def add_capitals(word):
    return ' '.join([w[0].upper() + w[1:] for w in word.split()])


def get_all_rappers(objects=False, remove_punc=True, wikia_spelling=False,
                    as_dict=False):
    if remove_punc:
        assert not objects

    if wikia_spelling:
        artists = db.artists.find({'wikia_spelling': {'$exists': True,
                                                      '$ne': ''}}, no_cursor_timeout=True)
    else:
        artists = db.artists.find(no_cursor_timeout=True)

    if remove_punc:
        attr = 'name'
        if wikia_spelling:
            attr = 'wikia_spelling'
        if as_dict:
            names = {remove_punctuation(a[attr]): a['name'] for a in artists}
        else:
            names = set([remove_punctuation(a[attr]) for a in artists])
        return names
    elif not objects:
        attr = 'name'
        if wikia_spelling:
            attr = 'wikia_spelling'
        if as_dict:
            names = {a[attr]: a['artist_id'] for a in artists}
        else:
            names = set([a[attr] for a in artists])
        return names
    else:
        return artists


def form_url(path):
    return "http://lyrics.wikia.com" + path


def wait():
    do_wait = np.random.random() > .95
    if do_wait:
        print "WAITING 1 MINUTE"
        sys.stdout.flush()
        # wait 1 minute
        time.sleep(60)
        print "DONE"
        sys.stdout.flush()


def get_artist_page(artist_name):
    url = form_url('/wiki/' + artist_name)
    wait()
    r = requests.get(url)
    return r.text


def scrape_song(song_url, current_rapper):
    url = form_url(song_url)
    if url.find('?action=edit') > -1:
        return None
    wait()
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        lyricbox = soup.find_all('div', 'lyricbox')[0]
    except Exception:
        return None
    else:
        verses = []
        verse = []
        prev_tag = False
        for child in lyricbox.descendants:
            if isinstance(child, Tag):
                if prev_tag:
                    verses.append(verse)
                    verse = []
                prev_tag = True
            else:
                line = unicode(child)
                verse.append(line)
                prev_tag = False
        new_verses = []
        for i, verse in enumerate(verses):
            if len(verse) == 0:
                continue
            elif len(verse) == 1:
                verse_type = type_of_verse(verse[0])
                nrps = extract_nrps(verse[0], current_rapper)
                new_verse = []
                if verse_type:
                    new_verse.append(verse_type)
                if nrps:
                    new_verse.append(nrps)
                if len(new_verse) > 0:
                    new_verses.append(new_verse)
                else:
                    new_verses.append(verse)
            else:
                verse_type = type_of_verse(verse[0])
                nrps = extract_nrps(verse[0], current_rapper)
                first_line_new_verse = []
                if verse_type:
                    first_line_new_verse.append(verse_type)
                if nrps:
                    first_line_new_verse.append(nrps)
                if len(first_line_new_verse) > 0:
                    new_verses.append(first_line_new_verse)
                    new_verses.append(verse[1:])
                else:
                    new_verses.append(verse)
        return new_verses


def type_of_verse(line):
    verse_type_words = [
        'prechorus',
        'chorus',
        'bridge',
        'verse',
        'breakdown',
        'hook',
        'intro',
        'outro',
        'introduction',
    ]
    for vt in verse_type_words:
        if remove_punctuation(line).find(vt) > -1:
            return "(SOV: {})".format(vt)


def translate_unicode(to_translate, translate_to=u' '):
    punc = [p for p in string.punctuation
            if p != '$']
    translate_table = dict((ord(char), translate_to) for char in unicode(punc))
    translate_table[ord(u'$')] = u's'
    return to_translate.translate(translate_table)


def extract_nrps(line, current_rapper):
    line = remove_punctuation(line)
    wikia_spelling = current_rapper['wikia_spelling']
    current_rapper_no_punc = remove_punctuation(wikia_spelling)

    rappers = []
    for rapper in [current_rapper_no_punc] + RAPPER_NO_PUNC_DICT.keys():
        rapper_index = line.find(rapper)
        if rapper_index > -1:
            # start of line or new word
            found_rapper = False
            if rapper_index == 0 or line[rapper_index - 1] == ' ':
                # end of line
                if rapper_index + len(rapper) >= len(line):
                    found_rapper = True
                # space on right side of rapper, new word
                elif line[rapper_index + len(rapper) + 1] == ' ':
                    found_rapper = True

            if found_rapper:
                rappers.append(RAPPER_NO_PUNC_DICT[rapper])
                line = line.replace(rapper, '')
        if len(line.strip()) < 2:
            break
    return ' '.join(['(NRP: {})'.format(r) for r in rappers])


def scrape_artist(doc):
    wikia_spelling = doc['wikia_spelling']
    html = get_artist_page(wikia_spelling)
    soup = BeautifulSoup(html, "html.parser")
    albums = soup.find(id="mw-content-text")

    all_albums = {}
    current_album = None
    current_album_year = None
    name_year_pattern = re.compile(r'(.*) *\(([0-9]{4})\)$')
    album_index = 0
    unnamed_index = 0
    try:
        albums.children
    except Exception:
        pass
    else:
        for child in albums.children:
            if not isinstance(child, Tag):
                continue
            child_albums = child.find_all('span', 'mw-headline')
            if len(child_albums):
                album_name = child_albums[0].text
                m = re.match(name_year_pattern, album_name)
                album_year = None
                if m:
                    album_name = m.group(1).strip()
                    album_year = int(m.group(2))
                current_album = album_name
                current_album_year = album_year
            elif child.name in ['ol', 'ul']:
                songs = {}
                for song_index, a in enumerate(child.find_all('a')):
                    song_url = a.get('href')
                    song_name = a.text
                    songs[str(song_index)] = {'name': song_name,
                                              'url': song_url}
                album_doc = {
                    'name': current_album,
                    'year': current_album_year,
                    'songs': songs,
                    'album_id': album_index
                }
                if current_album is None:
                    current_album = "Unnamed_{}".format(unnamed_index)
                    unnamed_index += 1
                all_albums[str(album_index)] = album_doc
                album_index += 1
                current_album = None
                current_album_year = None

        try:
            save_artist_albums_to_db(doc['artist_id'], all_albums)
        except Exception:
            pass
        else:
            for album_index, album in all_albums.iteritems():
                # first_album_index = album_index
                for song_index, song in album['songs'].iteritems():
                    # first_song_index = song_index
                    url = song['url']
                    print "TRYING URL: {}".format(url)
                    update_projection = {'artist_id': doc['artist_id']}
                    key = 'albums.{}.songs.{}.lyrics'.format(album_index,
                                                             song_index)
                    existing_lyrics = db.artists.find_one(update_projection,
                                                          {key: True})
                    if key in existing_lyrics and existing_lyrics[key]:
                        print "EXISTING LYRICS"
                        continue
                    try:
                        lyrics = scrape_song(url, doc)
                    except Exception:
                        pass
                    else:
                        if lyrics:
                            sys.stdout.flush()
                            print "SCRAPED LYRICS FOR URL: {}".format(url)
                            update_doc = {'$set': {key: lyrics}}
                            try:
                                db.artists.update(update_projection, update_doc)
                            except Exception:
                                pass


def save_artist_albums_to_db(artist_id, albums):
    db.artists.update({'artist_id': artist_id}, {'$set': {'albums': albums}})


def check_spellings(possible):
    for spelling in possible:
        html = get_artist_page(spelling)
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find(id="mw-content-text")
        if len(article.find_all('div', 'noarticletext')) > 0:
            continue
        else:
            return spelling


def find_artist_names():
    for i, rapper_obj in enumerate(get_all_rappers(objects=True,
                                                   remove_punc=False)):
        if 'wikia_spelling' in rapper_obj and\
                rapper_obj['wikia_spelling'] != '':
            print "ALREADY HAVE ENTRY FOR {}".format(rapper_obj['name'])
            continue

        rapper = rapper_obj['name']
        possible = [rapper]
        replace_spaces = rapper.replace(' ', '_')
        if replace_spaces not in possible:
            possible.append(replace_spaces)

        dashes = rapper.replace(' ', '-')
        if dashes not in possible:
            possible.append(dashes)

        no_punc = add_capitals(remove_punctuation(rapper))
        no_punc_spaces = no_punc.replace(' ', '_')
        if no_punc_spaces not in possible:
            possible.append(no_punc_spaces)
        no_punc_dashes = no_punc.replace(' ', '-')
        if no_punc_dashes not in possible:
            possible.append(no_punc_dashes)

        correct_spelling = check_spellings(possible)
        if correct_spelling:
            print "==>FOUND CORRECT SPELLING FOR {}".format(rapper)
            db.artists.update({'artist_id': rapper_obj['artist_id']},
                              {'$set': {'wikia_spelling': correct_spelling}})
        else:
            print "!COULD NOT FIND ENTRY FOR {}".format(rapper)
            db.artists.update({'artist_id': rapper_obj['artist_id']},
                              {'$set': {'wikia_spelling': ''}})


def find_no_wikia():
    artists = db.artists.find({'wikia_spelling': ''},
                              {'popularity': True, 'name': True})
    for artist in artists:
        if artist['popularity'] > 30:
            print "{}: {}".format(artist['name'], artist['popularity'])


def scrape_all_rappers(n=-1):
    rappers = get_all_rappers(wikia_spelling=True, objects=True,
                              remove_punc=False)
    print len([r for r in rappers])
    # rappers.batch_size(25)
    # i = 0
    # existing_rappers = []
    # for rapper in tqdm(rappers):
        # if 'albums' in rapper:
            # existing_rappers.append(rapper['artist_id'])
            # continue
        # scrape_artist(rapper)
        # sys.stdout.flush()
        # if i == n - 1:
            # return
        # i += 1
    # print existing_rappers

if __name__ == '__main__':
    global client
    global db
    client = MongoClient()
    db = client['lil-neuron-db']
    global RAPPER_NO_PUNC_DICT
    RAPPER_NO_PUNC_DICT = get_all_rappers(remove_punc=True,
                                          wikia_spelling=True,
                                          as_dict=True)
    scrape_all_rappers()
