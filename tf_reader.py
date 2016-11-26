import tensorflow as tf
from extract_features import RapFeatureExtractor
import pdb
import sys

# raw_data_features should be a list of 2D np arrays,
# where first dimension is features,
# second dimension is words
# third dimension is characters or phonemes


def batched_data_producer(extractor, batch_size, filename, num_epochs=None, name=None):
    tensor_dict, init_op_local = extractor.read_and_decode_single_example(filename, num_epochs=num_epochs)
    keys = tensor_dict.keys()
    values = tensor_dict.values()
    batched_data = tf.train.batch(
        tensors=values,
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="batched_features"
    )
    batch = {k: batched_data[i] for i, k in enumerate(keys)}
    return batch, init_op_local


def num_batches(extractor, batch_size, fname):
    batched, init_op_local = batched_data_producer(extractor, batch_size, fname, num_epochs=1)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config) as sess:
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


def run_and_return_one_batch(extractor, batch_size, fname):
    batched, init_op_local = batched_data_producer(extractor, batch_size, fname, num_epochs=1)

    with tf.Session() as sess:
        # Initialize the the epoch counter
        sess.run(init_op_local)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch = sess.run(batched)
        coord.request_stop()
        coord.join(threads)
    return batch

if __name__ == '__main__':

    extractor = RapFeatureExtractor(train_filenames=[],
                                    valid_filenames=[],
                                    from_config=True,
                                    config_file='data/config_test.p')
    batch_size = 1
    fname = 'data/tf_train_data_test.txt'
    print extractor.special_symbols
    print extractor.char_vocab_length
    batch = run_and_return_one_batch(extractor, 4, fname)
    pdb.set_trace()
