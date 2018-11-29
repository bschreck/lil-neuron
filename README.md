# Lil Neuron, the illest rapper alive

## A Recurrent Neural Network Language Model to Generate Rap Lyrics
Scrapes artists and metadata from Spotify, and lyrics from lyrics.wiki.com.
Processes lyrics to include spellchecking and pronunciations, including of slang, non-dictionary words (the hardest part)
Builds and trains RNNs combining word-level (via embeddings), character-level, pronunciations, syllabic stresses, and meta-contextual data.
Generates new lyrics from a seed.

# Explanation of Subdirectories and System Parts

## spotify_scraper
### Genre/Artist Scraper
Scrapes metadata about artists on Spotify. Starts with a seed list of artists, then recursively finds related artists. Saves to MongoDB.
Why is this written in Javascript/Coffeescript and not Python like the rest of the project? I don't know. I wrote this part 2 years ago and forgot. Might be easier to initialize Mongo models with Mongoose as opposed to Pymongo.

Files:
 * scraper.coffee
 * run_scraper.js
 * artist.coffee
 * genre.coffee

## data_viz
### Data Visualization
Generates histograms of various features of the data (collected from Spotify). For instance, the number of followers, or the number of plays.

Files:
 * db_viz.py

## spotify_scraper
### Lyrics Scraper
From the names of artists found on Spotify via the Genre/Artist Scraper, scrapes lyrics.wikia.com for all their lyrics.
Also does initial preprocessing of lyrics to find the names of artists on each verse, and metadata such as the song name, album, year.
Saves lyrics to MongoDB
Files:
 * scraper.py


## pronciation_annotator
### Pronunciation Annotator
Utility tool to easily and quickly annotate pronunciations of slang words. Breaks words into syllables, and pattern-matches syllables from new words with existing syllabic annotations. Deals effectively with plural and other forms of words.
 * find_pronunciations.py

## rnn_model
### Spell Checker
Tries to replace misspelled words. Tries swapping, adding, & removing vowels, and adding "g" to ing. Then searches for most probable word (by frequency in English or in a given corpus). Also implemented is finding words that are 1 or 2 away by editdistance, but I found that produced too many erroneous corrections.
Files:
 * spell_checker.py

### Lyrics Processor
Generates text files from lyrics in MongoDB. Tokenizes words using StanfordTokenizer, pulls pronunciations from each word using combination of
SpellChecker, CMU Pronouncing Corpus, and manually annotated pronunciations.

Files:
 * generate_lyric_files.py

### Word Vectors
Unfinished module to find which words have word embeddings in GloVe or Word2Vec.
Files:
* find_word_vectors.py

### Feature Extractor
Generates the actual input data to the deep network. Pulls phonemes and stresses from pronunications, separates words into characters, and generates context vectors for the metadata (e.g. artist, genre, album, year, etc)
Metadata is saved via rapper_matrix.py
Files:
 * extract_features.py
 * rapper_matrix.py

### RNN Model
Trains model and generates new lyrics. Generates embeddings for words first. Separate networks for words, characters, phonemes, stresses and context. Context network is non-recurrent.
Possibility to preload GloVe or Word2Vec vectors into the embeddings for known words as some initial semantic meaning. Not currently tested.
Files:
 * tf_reader.py
 * tf_rnn_model.py

## To Tokenize Corpus

Ensure you set the CLASSPATH environment variable to the
stanford-postagger.jar

    export CLASSPATH=stanford-postagger-2015-12-09/stanford-postagger.jar



