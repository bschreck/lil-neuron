mongoose = require 'mongoose'
deepPopulate = (require 'mongoose-deep-populate') mongoose
Schema = mongoose.Schema

ArtistSchema = new Schema
    artist_id: { type: String, index: true }
    name: String
    popularity: Number
    followers: Number
    related: [{type : String}]
    genres: [{type : mongoose.Schema.ObjectId, ref : 'Genre'}]
#ArtistSchema.plugin deepPopulate, {}
#ArtistSchema.methods.toFrontEnd = ->
#    artist_id: this.artist_id
#    name: this.name
#    popularity: this.popularity
#    followers: this.followers
#    genres: (genre.name for genre in this.genres)
module.exports = mongoose.model 'Artist', ArtistSchema
