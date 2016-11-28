import tensorflow as tf
from tensorflow.python.framework import ops
from extract_features import RapFeatureExtractor
import pdb
import sys

# raw_data_features should be a list of 2D np arrays,
# where first dimension is features,
# second dimension is words
# third dimension is characters or phonemes


def batched_data_producer(extractor, batch_size, max_num_steps, filename, num_epochs=None, name=None):
    tensor_dict, init_op_local = extractor.read_and_decode_single_example(max_num_steps, from_filename=filename, num_epochs=num_epochs)
    keys = tensor_dict.keys()
    values = tensor_dict.values()
    batched_data = tf.train.batch(
        tensors=values,
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="batched_features"
    )
    actual_batch_size = tf.shape(batched_data[0])[0]
    # TODO: figure out new verse_lengths per example in batch
    # and word_lengths per word per example in batch
    # each batch is
    # (for labels)
    # [1, 4, 3,    ..., 0 0 0 ],
    # [3, 2, 3, 5, ..., 8 9 2 ],
    # [4, 4,       ..., 0 0 0 ],
    # becomes
    # [1, 4] -> 2
    # [3, 2] -> 2
    # [4, 4] -> 2
    # and then
    # [3, 0] -> 1
    # [3, 5] -> 2
    # [0, 0] -> 0
    # verse_length_i_split = min(max_num_steps, max_num_steps - min(max_num_steps, verse_length_index_max - verse_length_i))
    to_slice_keys_3d = ["phones", "stresses", "chars"]
    verse_length = tf.shape([batched_data[i] for i in xrange(len(batched_data)) if keys[i] == 'phones'][0])[1]
    to_slice_3d = [tf.reshape(v, [verse_length, actual_batch_size, tf.shape(v)[2]])
                   for i, v in enumerate(batched_data) if keys[i] in to_slice_keys_3d]
    to_slice_keys_2d = ["phones.lengths", "stresses.lengths", "chars.lengths", "labels"]
    to_slice_2d = [tf.reshape(v, [verse_length, actual_batch_size])
                   for i, v in enumerate(batched_data) if keys[i] in to_slice_keys_2d]
    sliced_input, index_max = slice_n_input_producer(to_slice_3d + to_slice_2d, n=max_num_steps,
                                                     capacity=128,
                                                     shuffle=False, num_epochs=None)

    sliced_input_3d = [tf.reshape(v, [actual_batch_size, tf.shape(sliced_input[0])[0], -1])
                       for v in sliced_input[:len(to_slice_keys_3d)]]
    sliced_input_2d = [tf.reshape(v, [actual_batch_size, -1])
                       for v in sliced_input[len(to_slice_keys_3d):]]
    verse_lengths_i = keys.index('verse_length')
    verse_lengths = batched_data[verse_lengths_i]
    verse_lengths_split = tf.minimum(max_num_steps,
                                     max_num_steps - tf.minimum(max_num_steps,
                                                                index_max - verse_lengths))

    batched_data[verse_lengths_i] = verse_lengths_split
    batch = {k: sliced_input_3d[i] for i, k in enumerate(to_slice_keys_3d)}
    batch.update({k: sliced_input_2d[i] for i, k in enumerate(to_slice_keys_2d)})
    batch.update({k: batched_data[i] for i, k in enumerate(keys)
                  if k not in batch})

    init_op_local2 = tf.initialize_local_variables()
    return batch, init_op_local, init_op_local2


def slice_n_input_producer(tensor_list, n=1, shuffle=True, num_epochs=1,
                           seed=None, capacity=32,
                           shared_name=None, name=None):
    with tf.name_scope(name, "input_producer", tensor_list):
        tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
        if not tensor_list:
            raise ValueError(
                "Expected at least one tensor in slice_input_producer().")

        rs_queue = tf.train.input_producer([tf.shape(tensor_list[0])], shuffle=False,
                                           num_epochs=None, capacity=capacity)
        rs_tensor = rs_queue.dequeue()
        range_size = rs_tensor[0]
        queue = tf.train.range_input_producer(range_size, num_epochs=num_epochs,
                                              shuffle=shuffle, seed=seed, capacity=capacity,
                                              shared_name=shared_name)

        indices = queue.dequeue_up_to(n)
        index_max = indices[-1]
        output = [tf.gather(t, indices) for t in tensor_list]
        return output, index_max

def num_batches(extractor, batch_size, max_num_steps, fname):

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config) as sess:
        batched, init_op_local, init_op_local2 = batched_data_producer(extractor, batch_size, max_num_steps, fname, num_epochs=1)
        # Initialize the the epoch counter
        sess.run([init_op_local, init_op_local2])

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num_batches = 0
        try:
            while not coord.should_stop():
                # Retrieve a single instance:
                b = sess.run(batched)
                num_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
    print "num_batches:", num_batches
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
