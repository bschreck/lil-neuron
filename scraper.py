from bs4 import BeautifulSoup
from bs4 import Tag
import requests
import string
import pdb
"""
    Scrape from lyrics.wikia.com, after first scraping the artists to use from the top N rappers on
    last.fm, spotify, or echo nest, turn the artist's name into the format lyrics.wikia.com uses,
    go to the artist page there, and find the CSS class that lists all the songs (or albums) and urls
    Each song also includes an Artist: at the start of each verse if the artist is not the same as the
    artist who made the song, or who is a sub-artist (if it's a rap group)
"""
NO_PUNC_RAPPERS = ['jayz', 'lilkim']
def form_url(path):
    return "http://lyrics.wikia.com"+path
def get_artist_page(artist_name):
    url = form_url('/wiki/'+artist_name)
    r = requests.get(url)
    return r.text
def scrape_song(song_url, current_rapper):
    url = form_url(song_url)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        lyricbox = soup.find_all('div', 'lyricbox')[0]
    except:
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
            if len(verse) == 1:
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
        'verse'
        'breakdown'
    ]
    for vt in verse_type_words:
        if line.lower().find(vt) > -1:
            return vt

def translate_unicode(to_translate, translate_to=u' '):
    translate_table = dict((ord(char), translate_to) for char in unicode(string.punctuation))
    return to_translate.translate(translate_table)

def extract_nrps(line, current_rapper):
    line = translate_unicode(line.lower()).replace(' ','')
    current_rapper = translate_unicode(current_rapper.lower()).replace(' ','')

    rappers = []
    for rapper in [current_rapper] + NO_PUNC_RAPPERS:
        if line.find(rapper) > -1:
            rappers.append(rapper)
            line = line.replace(rapper, '')
        if len(line.strip()) < 2:
            break
    return ' '.join(['(NRP: {})'.format(r) for r in rappers])

def scrape_artist(artist_name):
    html = get_artist_page(artist_name)
    soup = BeautifulSoup(html, "html.parser")
    albums = soup.find(id="mw-content-text")
    artist_urls = []
    for ol in albums.find_all('ol'):
        for a in ol.find_all('a'):
            song_url = a.get('href')
            artist_urls.append(song_url)
    for song_url in artist_urls:
        lyrics = scrape_song(song_url)
    print len(artist_urls)
#scrape_artist('jay-z')
song = scrape_song('/wiki/Jay-Z:I_Know_What_Girls_Like', u'Jay-Z')
for verse in song:
    print "="*100
    for line in verse:
        print line


'''
Types of NRP:
(NRP)
(NRP), (NRP)
Chorus: NRP alsdkjfaldskj
Verse One: NRP


rapper names:
case sensitive
spaces become underscores
'''
