import tensorflow as tf
from extract_features import RapFeatureExtractor
import pdb


# TODO: in future try to have batches be different verses
def batched_data_producer(extractor, batch_size, max_num_steps, filename, num_epochs=None, capacity=32, name=None):
    tensor_dict, init_op_local = extractor.read_and_decode_single_example(max_num_steps, from_filename=filename, num_epochs=num_epochs)
    keys_to_batch = ['labels', 'phones', 'chars', 'stresses', 'phones.lengths', 'chars.lengths', 'stresses.lengths']
    to_batch = {k: v for k, v in tensor_dict.iteritems() if k in keys_to_batch}

    verse_length = tensor_dict.pop('verse_length')
    context_features = [k for k in tensor_dict if k not in keys_to_batch]
    for c in context_features:
        multiples = tf.pack([verse_length[0], 1])
        to_batch[c] = tf.tile(tf.expand_dims(tensor_dict[c], 0),
                              multiples)
    batch1 = tf.train.batch(
        tensors=to_batch,
        enqueue_many=True,
        batch_size=max_num_steps,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="num_steps_batch"
    )
    batch2 = tf.train.batch(
        tensors=batch1,
        enqueue_many=False,
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=False,
        name="batch_size_batch"
    )
    init_op_local2 = tf.local_variables_initializer()
    return batch2, init_op_local, init_op_local2


def num_batches(extractor, batch_size, max_num_steps, fname):
    def inner_func(batch, res):
        if res is None:
            return 1, False
        else:
            return res + 1, False
    return run_one_epoch(inner_func, extractor, batch_size, max_num_steps, fname)

def run_and_return_batches(extractor, num_batches, batch_size, max_num_steps, fname):
    def inner_func(batch, res):
        should_stop = False
        if res is None:
            if num_batches == 1:
                should_stop = True
            return [batch], should_stop
        else:
            res.append(batch)
            if len(res) == num_batches:
                should_stop = True
            return res, should_stop
    return run_one_epoch(inner_func, extractor, batch_size, max_num_steps, fname)


def run_one_epoch(inner_func, extractor, batch_size, max_num_steps, fname):
    with tf.Graph().as_default():
        batched, init_op_local, init_op_local2 = batched_data_producer(extractor, batch_size, max_num_steps, fname, num_epochs=1)

        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=20)

        initializer = tf.global_variables_initializer()
        with tf.Session(config=tf_config) as sess:
            # Initialize the the epoch counter
            sess.run([initializer, init_op_local, init_op_local2])

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            res = None
            try:
                while not coord.should_stop():
                    # Retrieve a single instance:
                    b = sess.run(batched)
                    res, should_stop = inner_func(b, res)
                    if should_stop:
                        break
            except tf.errors.OutOfRangeError:
                pass
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
        return res


if __name__ == '__main__':
    extractor = RapFeatureExtractor(train_filenames=[],
                                    valid_filenames=[],
                                    from_config=True,
                                    config_file='data/config_new.p')
    batch_size = 2
    max_num_steps = 100
    fname = 'data/tf_train_data_new.txt'
    batches = run_and_return_batches(extractor, 100, batch_size, max_num_steps, fname)
    pdb.set_trace()
