# Lil Neuron, the illest rapper alive

Lil Neuron don't need DNA

to drink Alizé,

make out your girl erryday

sell out stadiums and make more

rap lyrics



Lil Neuron picks words so quick

you gotta predict he was not intermixed

from the sloppy old play of chicks and dicks

rather

just assembled

mesh of old lyrics in order to resemble

old school rappers in the flesh and in the blood

running on electrons

gunning for respect on

the highest stage

in effect direct rejection

of the longest evolution

to be the one and only

rapper perfected

## A Recurrent Neural Network Language Model to Generate Rap Lyrics
IDEA: use pretrained RNN (trained on standard english), and then fine
tune here with rap lyrics

Training Model:
 1. Create preprocessor to create several different representations of
    each song using words, pronunciations, rhyme schemes, and stresses
 2. Each feature set will be split up with groups of various layers,
    like phoneme_level -> word_level -> phrase_level -> verse_level
 3. Preprocessing will assign a symbol to each break in the levels,
    so there will be a symbol for the next phoneme, the next word,
    the next phrase, the next verse, end of song, etc.
 4. Songs will then end up being transformed into several vectors of
    these symbols, one for each feature. Each feature vector will be padded
    to the same size
 5. Feature vectors will be fed into separate RNN networks.
 6. The first layer is a one-hot encoding of each word into an
    n-dimensional vector, where n is the total number of words in the
    corpus (or phonemes depending on the feature). This layer is not
    learned.
 7. The second layer is an embedding into a lower dimensional feature
    space of size 100-300
 8. The third and fourth layers are LSTM or GRU's, interweaved with
    Dropout layers
 9. The fifth layer combines the output from each of the fourth layers
    from each feature and concatenates them
 10. The sixth layer is a strongly connected layer that converts the
     LSTM-outputs into a probability vector over the vocabulary size
 11. Update gradients using softmax/cross-entropy over this output vector,
     compared to the actual next word in the song/corpus

Generating Model:
 1. Come up with the first few phrases
 2. Run the trained model without backprop to generate the rest of the
    song. Model stops at end-of-song symbol

# To Tokenize Corpus

Ensure you set the CLASSPATH environment variable to the
stanford-postagger.jar

    export CLASSPATH=stanford-postagger-2015-12-09/stanford-postagger.jar


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

