mongoose = require 'mongoose'
Schema = mongoose.Schema

GenreSchema = new Schema
    genre_id: { type: String, index: true }
    name: String
module.exports = mongoose.model 'Genre', GenreSchema
