mongoose = require 'mongoose'
uniqueValidator = require 'mongoose-unique-validator'
Schema = mongoose.Schema

GenreSchema = new Schema
    name:
        type: String
        required: true
        unique: true
GenreSchema.plugin uniqueValidator
module.exports = mongoose.model 'Genre', GenreSchema
