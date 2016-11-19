import tensorflow as tf
# [0, 1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")

# A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

# Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
y = tf.slice(x, [0], [slice_end], name="y")
z = tf.slice(x, [0], [slice_end+1], name="z")

# Batch the variable length tensor with dynamic padding
batched_data = tf.train.batch(
    tensors=[y,z],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"y": batched_data}, n=2, feed_dict=None)

# Print the result
#print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])
print(res[1]["y"])
