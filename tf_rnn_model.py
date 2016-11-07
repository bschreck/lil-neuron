import tensorflow as tf
# Use dynamic_rnn()
# dynamically pad batches using tf.train.batch
# dynamic_rnn allows for a changing batch_size, so my batch_size can be the length of the verse
# input sequence_length of non-padded lengths of each sentence/line

# multiple dynamic_rnn's for each sequence feature
# normal NN's for each context feature
# combine them and score

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, dynamic_batch_size=True, name=None):
        self.batch_size = None
        if not dynamic_batch_size:
            self.batch_size = config.batch_size
            self.input_data, self.targets, self.sequence_lengths = \
                reader.extract(data, batch_size, name=name)
            self.epoch_size = self.find_ep
        else:
            self.batch_size = config.batch_size
            self.epoch_size = ((len(data) // self.batch_size) - 1) // num_steps
            self.input_data, self.targets, self.sequence_lengths, self.dynamic_batch_size = \
                reader.extract(data, batch_size = None, name=name)
    @property
    def epoch_size(self):
        pass
        # TODO: figure this out
        #self.epoch_size = ((len(data) // self.batch_size) - 1) // num_steps


class RNNPath(object):
    def __init__(self, is_training, config, input_, device="/cpu:0"):
        self.device = device
        self._input = input_
        batch_size = input_.batch_size
        X_lengths = input_.sequence_lengths
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Can experiment with different settings
        lstm_cell = tf.nn.rnn_cell.LSTMCell(size,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=1,
               num_proj_shards=1,
               forget_bias=1.0,
               state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                  lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers,
                                           state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())


        with tf.device(device):
            inputs = rf.rnn_cell._linear(input_.input_data,
                                         size,
                                         False)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                 dtype=data_type(),
                                                 sequence_length=X_lengths,
                                                 inputs=inputs)
        self._final_state = last_states[-1]
        self._outputs = outputs
    @property
    def input(self):
      return self._input

    @property
    def initial_state(self):
      return self._initial_state

    @property
    def final_state(self):
      return self._final_state

    @property
    def outputs(self):
      return self._outputs

class ConcatLearn(object):
    def __init__(self, is_training, config, input_, device="/cpu:0"):
        self.rnn_paths = input_.rnn_paths
        self.feed_forward_path = input_.feed_forward_path

        combined_outputs = zip(*[path.outputs for path in self.rnn_paths])
        concat_outputs = tf.concat(-1, combined_outputs)
        concat_final_shape = concat_outputs[0].shape[-1]
        outputs_flat = tf.reshape(concat_outputs, [-1, concat_final_shape])

        outputs_flat = tf.concat(-1, [self.feed_forward_path.outputs, outputs_flat])
        final_unit_size = outputs_flat[0].shape[-1]



        # Output layer weights
        softmax_W = tf.get_variable(
            name="softmax_w",
            initializer=tf.random_normal_initializer(),
            shape=[final_unit_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        logits_flat = tf.batch_matmul(outputs_flat, softmax_W) + softmax_b
        #probs_flat = tf.nn.softmax(logits_flat)

        # Calculate the losses
        y_flat =  tf.reshape(input_.targets, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses

        # Bring back to [B, T] shape
        masked_losses = tf.reshape(masked_losses,  tf.shape(input_.targets))

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / X_lengths
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        self._cost = cost = mean_loss

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    @property
    def input(self):
        return self._input
    @property
    def cost(self):
        return self._cost
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000



