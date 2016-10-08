request = require 'request'
fs = require 'fs'
# Set the configuration settings
credentials =
    client:
        id: '94f83bdec0424c83bfc78ed77f3bd82b'
        secret: '6fb1703665f04b39ae1ccdc8c4064c07'

    auth:
        tokenHost: 'https://accounts.spotify.com'
        tokenPath: '/api/token'

oauth2 = require('simple-oauth2').create(credentials)
module.exports.getArtists = (Genre, Artist)->
    tokenConfig = {}
    # Callbacks
    # Get the access token object for the client
    oauth2.clientCredentials.getToken tokenConfig, (error, result) ->
        if (error)
            return console.log('Access Token Error', error.message)
        token = oauth2.accessToken.create(result)
        console.log "token:", token.token.access_token

        getRequestOptions = (token, url) ->
            header =
                "Authorization": "Bearer #{token.token.access_token}"
            options =
                url: url
                headers: header
            options
        getArtist = (token, base_url, artist_id, callback)->
            artist_url = "/artists/#{artist_id}"
            url = base_url + artist_url
            options = getRequestOptions(token, url)
            request.get options, (error, response, body) ->
                if (error)
                    callback error.message
                callback JSON.parse(body)
        getRelatedArtists = (token, base_url, artist_object_id, artist_id, callback)->
            related_url = "/artists/#{artist_id}/related-artists"
            url = base_url + related_url
            options = getRequestOptions token, url
            request.get options, (error, response, body) ->
                if (error)
                    return console.log('Related Artist Error', error.message)
                artists = JSON.parse(body)["artists"]
                if artists?
                    callback artist_object_id, artists
                else
                    callback artist_object_id, []


        getAllArtistsBody = (token) ->
            jayz_id = '3nFkdlSjzX9mRTtwJOzDYB'
            base_url = 'https://api.spotify.com/v1'
            i = 0
            getRelatedArtistCallback = (prev_artist_id, artists) ->
                i += 1
                if i > 1
                    return
                related = (artist.id for artist in artists)
                Artist.update({ _id: prev_artist_id }, { $set: { related: related }}).exec();
                for artist in artists
                    saveArtistToDB artist, (artist_obj) ->
                        getRelatedArtists token, base_url, artist_obj._id, artist.id, getRelatedArtistCallback
            first_artist = getArtist token, base_url, jayz_id, (artist) ->
                saveArtistToDB artist, (artist_obj)->
                    artist_id = artist.id
                    artist_object_id = artist_obj._id
                    getRelatedArtists token, base_url, artist_object_id, artist_id, getRelatedArtistCallback

        # TODO: move validators to within model, and set unique to True
        saveGenreToDB = (genre, next)->
            if genre == "crunk"
                console.log "CRUNK"
            Genre.findOne {name: genre}, (err, obj)->
                if genre == "crunk"
                    console.log "STATUS:", err, obj
                if obj?
                    next obj
                else
                    genre = Genre({name:genre})
                    genre.save (err)->
                        if err
                            next null
                        else
                            next genre
        saveArtistToDB = (artist, callback) ->
            if artist?
                if artist.name? and artist.popularity? and artist.id?
                    genres = artist.genres
                    genre_id = 0
                    genre_objs = []
                    genre_callback = (genre_obj)->
                        genre_objs.push genre_obj._id
                        genre_id += 1
                        if genre_id < genres.length
                            saveGenreToDB genres[genre_id], genre_callback
                        else
                            Artist.findOne {artist_id: artist.id}, (err, obj)->
                                if obj?
                                    callback obj
                                else
                                    artist_obj = Artist
                                        name: artist.name
                                        popularity: artist.popularity
                                        followers: artist.followers.total
                                        artist_id: artist.id
                                        genres: genre_objs
                                    artist_obj.save (err)->
                                        if err
                                            console.log "ERR SAVING ARTIST:",err
                                            callback null
                                        else
                                            callback artist_obj
                    saveGenreToDB genres[genre_id], genre_callback
        getAllArtistsBody(token)

