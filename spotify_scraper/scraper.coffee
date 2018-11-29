request = require 'request'
fs = require 'fs'

JAYZ_ID = '3nFkdlSjzX9mRTtwJOzDYB'
KENDRICK_ID = '2YZyLoL8N0Wb9xBt1NhZWg'
TRIBE_CALLED_QUEST_ID = '09hVIj6vWgoCDtT03h8ZCa'
GUCCI_MANE_ID = '13y7CgLHjMVRMDqxdx0Xdo'
YACHTY = '6icQOAFXDZKsumw3YXyusw'
WU_TANG = '34EP7KEpOjXcM2TCat1ISk'
RUN_THE_JEWELS = '4RnBFZRiMLRyZy0AzzTg2C'
MF_DOOM = '2pAWfrd7WFF3XhVt9GooDL'
LIL_DICKY = '1tqhsYv8yBBdwANFNzHtcr'
LIL_WAYNE = '55Aa2cqylxrFIXC767Z865'
TYLER_THE_CREATOR = '4V8LLVI7PbaPR0K2TGSxFF'
PUBLIC_ENEMY = '6Mo9PoU6svvhgEum7wh2Nd'
THEORY_HAZIT = '23yxO4nVI3C2CoXIkYLifD'
BEEDIE = '75haPGtpJ5ZotdoAg3FOTQ'
DUMFOUNDEAD = '7LTShHcq1KdTrWeLvWoYed'
JAZZ_ADDIXX = '7tPGimUaIMq6r5Qr8lzpSS'
LIL_KIM = '5tth2a3v0sWwV1C7bApBdX'
FIRST_ARTIST_ID = LIL_KIM
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
            base_url = 'https://api.spotify.com/v1'
            getRelatedArtistCallback = (prev_artist_id, artists) ->
                related = (artist.id for artist in artists)
                Artist.update({ _id: prev_artist_id }, { $set: { related: related }}).exec();
                for artist in artists
                    saveArtistToDB artist, (artist_obj) ->
                        if artist_obj?
                            getRelatedArtists token, base_url, artist_obj._id, artist.id, (_id, artists)->
                                setTimeout getRelatedArtistCallback, 500, _id, artists
            first_artist = getArtist token, base_url, FIRST_ARTIST_ID, (artist) ->
                saveArtistToDB artist, (artist_obj)->
                    if artist_obj?
                        artist_id = artist.id
                        artist_object_id = artist_obj._id
                        getRelatedArtists token, base_url, artist_object_id, artist_id, (prev_artist_id, artists)->
                            setTimeout getRelatedArtistCallback, 500, prev_artist_id, artists
                    else
                        console.log "Not retrieving related artists, already have DB entry for #{artist.name}"

        saveGenreToDB = (genre, next)->
            genre_obj = Genre({name:genre})
            genre_obj.save (err)->
                if err
                    Genre.findOne {name:genre}, (err, obj) ->
                        if obj?
                            next obj
                        else
                            console.log "ERR SAVING GENRE: #{genre_obj.name}; #{err}"
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
                        if genre_obj?
                            genre_objs.push genre_obj._id
                        genre_id += 1
                        if genre_id < genres.length
                            saveGenreToDB genres[genre_id], genre_callback
                        else
                            artist_obj = Artist
                                name: artist.name
                                popularity: artist.popularity
                                followers: artist.followers.total
                                artist_id: artist.id
                                genres: genre_objs
                            artist_obj.save (err)->
                                if err
                                    callback null
                                    ###
                                      #Artist.findOne {name:artist.name}, (err, obj) ->
                                      #    if obj?
                                      #        callback obj
                                      #    else
                                      #        console.log "ERR SAVING ARTIST: #{artist.name}; #{err}"
                                      #        callback null
                                      ###
                                else
                                    callback artist_obj
                    saveGenreToDB genres[genre_id], genre_callback
        getAllArtistsBody(token)

