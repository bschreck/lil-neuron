from pymongo import MongoClient
import os
import pdb


def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)


def generate_files(dirname):
    rappers = db.artists.find({'albums': {'$exists': True}})
    for rap_i, rapper in enumerate(rappers):
        rapper_name = rapper["name"]
        albums = rapper['albums']
        if type(albums) != list:
            iterator = albums.iteritems()
        else:
            iterator = enumerate(albums)

        for album_i, album in iterator:
            album_name = album["name"]
            album_year = album["year"]
            songs = album['songs']
            unknown_song_index = 0

            if type(songs) != list:
                song_iterator = songs.iteritems()
            else:
                song_iterator = enumerate(songs)
            for song_i, song in song_iterator:
                song_name = song["name"]
                if "lyrics" not in song or not song["lyrics"]:
                    continue
                lyrics = song["lyrics"]
                filename = formulate_filename(dirname,
                                              rapper_name,
                                              album_year,
                                              album_name,
                                              song_name)
                if filename.endswith("unknown_song"):
                    filename += str(unknown_song_index)
                    unknown_song_index += 1
                mkdir_recursive(os.path.dirname(filename))
                with open(filename, 'w') as f:
                    if not lyrics[0][0].startswith('(NRP:'):
                        f.write(u'(NRP: {})\n'.format(rapper_name).encode('utf8'))
                    for verse in lyrics:
                        f.write(u'\n'.join(verse).encode('utf8'))
                        f.write(u'\n<eov>\n'.encode('utf8'))


def formulate_filename(dirname, rapper, year, album, song):
    if not year:
        year = "unknown_year"
    if not album:
        album = "unknown_album"
    if not song:
        song = "unknown_song"
    return "/".join([dirname, rapper, str(year), album, song]) + ".txt"

def all_filenames(dirname):
    filenames = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


if __name__ == '__main__':
    global client
    global db
    client = MongoClient()
    db = client['lil-neuron-db']
    #generate_files("data/lyric_files")
    filenames = all_filenames("data/lyric_files")
    filenames = all_filenames("data/lyric_files/Tyler, The Creator")
