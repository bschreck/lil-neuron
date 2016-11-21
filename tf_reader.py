import tensorflow as tf
from extract_features import RapFeatureExtractor
import pdb
import sys

# raw_data_features should be a list of 2D np arrays,
# where first dimension is features,
# second dimension is words
# third dimension is characters or phonemes
def batched_data_producer(extractor, batch_size, filename, num_epochs=None, name=None):
    # TODO: num_steps?
    context, sequence, init_op_local = extractor.read_and_decode_single_example(filename, num_epochs=num_epochs)
    rapper0 = tf.cast(context['rapper0'], tf.int32)
    verse_length = tf.cast(context['verse_length'], tf.int32)
    word_length = tf.cast(context['word_length'], tf.int32)

    labels = tf.cast(sequence['labels'], tf.int32)

    phones_lengths = tf.cast(sequence['phones.lengths'], tf.int32)
    phones_shape = tf.cast(context['phones.shape'], tf.int32)
    phones = tf.cast(sequence['phones'], tf.int32)
    phones = tf.reshape(phones, phones_shape)

    chars_lengths = tf.cast(sequence['chars.lengths'], tf.int32)
    chars_shape = tf.cast(context['chars.shape'], tf.int32)
    chars = tf.cast(sequence['chars'], tf.int32)
    chars = tf.reshape(chars, chars_shape)

    stresses_lengths = tf.cast(sequence['stresses.lengths'], tf.int32)
    stresses_shape = tf.cast(context['stresses.shape'], tf.int32)
    stresses = tf.cast(sequence['stresses'], tf.int32)
    stresses = tf.reshape(stresses, stresses_shape)
    batched_data = tf.train.batch(
        tensors=[rapper0, labels,
                 chars, chars_lengths,
                 phones, phones_lengths,
                 stresses, stresses_lengths,
                 verse_length, word_length],
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="batched_features"
    )
    return {
            'rapper0': batched_data[0],
            'labels': batched_data[1],
            'chars': batched_data[2],
            'chars_lengths': batched_data[3],
            'phones': batched_data[4],
            'phones_lengths': batched_data[5],
            'stresses': batched_data[6],
            'stresses_lengths': batched_data[7],
            'verse_length': batched_data[8],
            'word_length': batched_data[9],
    }, init_op_local


def num_batches(extractor, batch_size, fname):
    batched, init_op_local = batched_data_producer(extractor, batch_size, fname, num_epochs=1)

    with tf.Session() as sess:
        # Initialize the the epoch counter
        sess.run(init_op_local)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num_batches = 0
        try:
            while not coord.should_stop():
                # Retrieve a single instance:
                sess.run(batched)
                num_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
    return num_batches

if __name__ == '__main__':

    extractor = RapFeatureExtractor(train_filenames=[],
                                    valid_filenames=[],
                                    from_config=True,
                                    config_file='data/config_test.p')
    batch_size = 1
    fname = 'data/tf_train_data_test.txt'
    print extractor.special_symbols
    print extractor.char_vocab_length
    #print num_batches(extractor, batch_size, fname, initialize=False)
