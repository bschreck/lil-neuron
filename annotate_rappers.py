from pymongo import MongoClient
import pdb


def artist_cycle(fn, ask=True, condition={}):
    client = MongoClient()
    db = client["lil-neuron-db"]
    artists = db.artists.find(condition)
    for artist in artists:
        name = artist["name"]
        print "$ RAPPER: {}".format(name)
        if ask:
            do_rating = raw_input(">> Rate? y/n")
        else:
            do_rating = 'y'
        if do_rating and do_rating.lower().startswith('y'):
            fn(artist, db)


def rate_songs():
    artist_cycle(rate_rapper_songs)


def rate_rapper(doc, db):
    rapper_name = doc["name"]
    got_rating = False
    while not got_rating:
        rating = raw_input("Rating (1-3): ")
        try:
            rating = int(rating)
        except:
            print "Invalid rating"
        else:
            got_rating = True
    save_rapper_rating(rapper_name, rating, db)


def rate_rapper_songs(doc, db):
    rapper_name = doc["name"]

    for album_i in sorted(doc['albums'].keys(),
                          key=lambda x: int(x)):
        album = doc['albums'][album_i]
        album_year = album.get('year')
        album_name = album.get('name')
        for song_i in sorted(album['songs'].keys(),
                key=lambda x: int(x)):
            song = album['songs'][song_i]
            name = song.get('name')
            lyrics = song.get('lyrics')
            if not lyrics:
                continue
            quality = song.get('quality')
            uniqueness = song.get('uniqueness')
            if quality is not None or uniqueness is not None:
                continue
            try:
                print "=== SONG: {} ({}, from {})===".format(name, album_year, album_name)
            except:
                continue
            print "Hit enter to pass, or type 'l' to see lyrics"
            quality = raw_input(">>> Song quality rating? 1-5")
            if quality == "l":
                try:
                    for l in lyrics:
                        print l
                except:
                    pass
                quality = raw_input(">>> Song quality rating? 1-5")
            quality = sanitize(quality)


            unique = raw_input(">>> Song uniqueness rating? 1-5")
            if unique == "l":
                try:
                    for l in lyrics:
                        print l
                except:
                    pass
                unique = raw_input(">>> Song uniqueness rating? 1-5")
            unique = sanitize(unique)
            if unique== "":
                unique = None
            if quality is not None or unique is not None:
                save_song_rating(rapper_name, album_i, song_i, quality, unique, db)

def sanitize(quality):
    if quality == "":
        quality = None
    else:
        while True:
            try:
                quality = int(quality)
            except:
                quality = raw_input(">>> Song quality rating? 1-5")
            else:
                if quality < 6 and quality > 0:
                    break
                else:
                    print "Must be lower than 6 and greater than 0"
    return quality
def save_song_rating(rapper_name, album_i, song_i, quality, unique, db):
    key_name_base = 'albums.{}.songs.{}'.format(album_i, song_i)
    update_doc = {
            }
    if quality is not None:
        key_name = key_name_base + ".quality"
        update_doc[key_name] = quality
    if unique is not None:
        key_name = key_name_base + ".uniqueness"
        update_doc[key_name] = unique
    if len(update_doc):
        db.artists.update_one({'name': rapper_name}, {'$set': update_doc})


def save_rapper_rating(rapper_name, rating, db):
    db.artists.update_one({'name': rapper_name}, {'$set': {'rating': rating}})


def save_rating_manual(rapper_name, album_name, song_name, quality, unique):
    client = MongoClient()
    db = client["lil-neuron-db"]
    artist = db.artists.find_one({"name": rapper_name})
    album_index = None
    for album_i in artist['albums']:
        album = artist['albums'][album_i]
        if album['name'] == album_name:
            album_index = album_i
            for song_i in album['songs']:
                song = album['songs'][song_i]
                if song['name'] == song_name:
                    save_song_rating(rapper_name, album_index, song_i, quality, unique, db)
                    return True


def find_quality_rappers():
    # TODO: generate list of quality rappers
    # rate on scale 1-3 with 3 meaning include the most copies
    # and 1 only include once
    condition = {'rating': {'$exists': False}}
    artist_cycle(rate_rapper, ask=False, condition=condition)


if __name__ == '__main__':
    find_quality_rappers()
    #rate_rap_songs()
    #song = "Siiiiiiiiilver Surffffeeeeer Intermission"
    #album = "The Life of Pablo"
    #success = save_rating_manual("Kanye West", album, song, 1, 1)
    #print success
