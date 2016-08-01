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
