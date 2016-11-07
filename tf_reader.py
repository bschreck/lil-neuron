import tensorflow as tf

# raw_data_features should be a list of 2D np arrays,
# where first dimension is features,
# second dimension is words
# third dimension is characters or phonemes
def batched_data_producer(raw_data_features, batch_size, num_steps, name=None):
    tensor_batched_x = []
    tensor_batched_y = []
    for i, raw_data_feature in enumerate(raw_data_features):
        with tf.name_scope(name, "BatchedDataProducer", [raw_data_feature, batch_size, num_steps]):
            raw_data = tf.convert_to_tensor(raw_data_feature, name="raw_data_f"+str(i), dtype=tf.int32)

            data_len = tf.size(raw_data)
            batch_len = data_len // batch_size
            data = tf.reshape(raw_data[0 : batch_size * batch_len],
                              [batch_size, batch_len, -1])

            epoch_size = (batch_len - 1) // num_steps
            assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
            with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.slice(data, [0, i * num_steps, 0], [batch_size, num_steps, -1])
            y = tf.slice(data, [0, i * num_steps + 1, 0], [batch_size, num_steps, -1])

            tensor_batched_x.append(x)
            tensor_batched_y.append(y)
            # # Creating a new queue
            # padding_q = tf.PaddingFIFOQueue(
                # capacity=2*batch_size,
                # dtypes=tf.int32,
                # shapes=[[None]])

            # # Enqueue the examples
            # enqueue_op = padding_q.enqueue([x])

            # # Add the queue runner to the graph
            # qr = tf.train.QueueRunner(padding_q, [enqueue_op])

            # # Creating a new queue
            # padding_qy = tf.PaddingFIFOQueue(
                # capacity=2*batch_size,
                # dtypes=tf.int32,
                # shapes=[[None]])

            # # Enqueue the examples
            # enqueue_opy = padding_qy.enqueue([x])

            # # Add the queue runner to the graph
            # qry = tf.train.QueueRunner(padding_qy, [enqueue_opy])
            # tf.train.add_queue_runner(qry)

            # # Dequeue padded data
            # batched_data_x = padding_q.dequeue_many(batch_size)
            # batched_data_y = padding_qy.dequeue_many(batch_size)
            # tensor_batched_x.append(batched_data_x)
            # tensor_batched_y.append(batched_data_y)
    return tensor_batched_x, tensor_batched_y
