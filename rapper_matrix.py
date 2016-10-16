from pymongo import MongoClient
import pdb
import cPickle as pck
import numpy as np


def build_rapper_matrix(pfile='rapper_matrix.p'):
    client = MongoClient()
    db = client["lil-neuron-db"]
    num_artists = db.artists.count()
    num_genres = db.genres.count()
    artists = db.artists.find()
    genres = db.genres.find()

    artist_indices = {}
    genre_indices = {}
    for i, artist in enumerate(artists):
        artist_indices[artist['artist_id']] = i
    for i, genre in enumerate(genres):
        genre_indices[str(genre['_id'])] = i

    rapper_matrix = {}
    artists = db.artists.find()
    for artist in artists:
        artist_vector = np.zeros(num_artists + num_artists + num_genres)
        name = artist['name']
        genres = artist['genres']
        related = artist['related']
        related_indices = [num_artists + artist_indices[a_id] for a_id in related
                           if a_id in artist_indices]
        current_genre_indices = [num_artists + genre_indices[str(g_id)] for g_id in genres
                                 if g_id and g_id != 'None']
        artist_vector[artist_indices[str(artist['artist_id'])]] = 1
        artist_vector[related_indices] = 1
        artist_vector[current_genre_indices] = 1
        rapper_matrix[name] = artist_vector
    pck.dump(rapper_matrix, open(pfile, 'wb'))
    return rapper_matrix


if __name__ == '__main__':
    rapper_matrix = build_rapper_matrix()
    first = rapper_matrix.keys()[0]
    print first
    print rapper_matrix[first]
