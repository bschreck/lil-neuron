import tensorflow as tf
import numpy as np
import time

import tf_reader as reader
from extract_features import RapFeatureExtractor

import pdb

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
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("train_filename", 'data/tf_train_data_test.txt',
                    "where the training data is stored.")
flags.DEFINE_string("valid_filename", 'data/tf_valid_data_test.txt',
                    "where the validation data is stored.")
flags.DEFINE_string("save_path", 'models',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class LNInput(object):
    """The input data."""

    def __init__(self, extractor, config, filename, name=None):
        self.extractor = extractor
        self.batch_size = config.batch_size
        self.filename = filename
        self.input_data, _ = reader.batched_data_producer(self.extractor, self.batch_size, self.filename, name=name)
        print "Retrieving epoch size"

        self._epoch_size = reader.num_batches(extractor, self.batch_size, self.filename)
        print self._epoch_size

    @property
    def epoch_size(self):
        return self._epoch_size

    def __getattr__(self, key):
        return self.input_data[key]

    def __getitem__(self, key):
        return self.input_data[key]




class RNNInput(object):
    def __init__(self, feature, word_lengths, sequence_lengths, verse_length, word_length):
        self.feature = feature
        self.word_lengths = word_lengths
        self.sequence_lengths = sequence_lengths
        self.verse_length = verse_length
        self.word_length = word_length

class RNNPath(object):
    def __init__(self, is_training, config, input_, device="/cpu:0"):
        self.device = device
        self._input = input_
        feature = input_.feature
        batch_size = config.batch_size
        char_vocab_size= config.char_vocab_size
        word_lengths = input_.word_lengths
        verse_lengths = input_.sequence_lengths
        feature_shape = tf.shape(feature)
        batch_verse_len = feature_shape[1]

        self.size = config.hidden_size

        # Can experiment with different settings
        lstm_args = dict(
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=1,
               num_proj_shards=1,
               forget_bias=1.0,
               state_is_tuple=True)
        char_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.size, **lstm_args)
        word_lstm_cell = tf.nn.rnn_cell.LSTMCell(2*self.size, **lstm_args)

        if is_training and config.keep_prob < 1:
            char_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                  char_lstm_cell, output_keep_prob=config.keep_prob)
            word_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                  word_lstm_cell, output_keep_prob=config.keep_prob)
        word_level_cell = tf.nn.rnn_cell.MultiRNNCell([word_lstm_cell] * config.num_word_layers,
                                           state_is_tuple=True)
        char_level_cell_fw = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell] * config.num_char_layers,
                                           state_is_tuple=True)
        char_level_cell_bw = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell] * config.num_char_layers,
                                           state_is_tuple=True)

        self._initial_states = [word_level_cell.zero_state(batch_size, data_type()),
                                char_level_cell_fw.zero_state(batch_size, data_type()),
                                char_level_cell_bw.zero_state(batch_size, data_type())]


        with tf.device(device):
            # [batch_size, verse_len, word_len]
            new_shape = tf.pack([batch_size * batch_verse_len, -1])
            input_data_concat = tf.reshape(feature, new_shape)
            # [batch_size * verse_len, word_len]

            embedding = tf.get_variable(
                "embedding", [char_vocab_size, self.size], dtype=data_type())
            embed_inputs = tf.nn.embedding_lookup(embedding, input_data_concat)
            # [batch_size * verse_len, word_len, size]

            word_lengths = tf.reshape(word_lengths, tf.pack([batch_size * batch_verse_len]))

            _, states = tf.nn.bidirectional_dynamic_rnn(char_level_cell_fw,
                                                        char_level_cell_bw,
                                                        embed_inputs,
                                                        sequence_length=word_lengths,
                                                        dtype=data_type(),
                                                        swap_memory=True,
                                                        time_major=False,
                                                        scope=None)
            fw_states, bw_states = states
            fw_state = [_s for fw_state in fw_states for _s in [fw_state.c, fw_state.h]]
            bw_state = [_s for bw_state in bw_states for _s in [bw_state.c, bw_state.h]]
            full_state = fw_state + bw_state

            both_states = tf.concat(1, full_state)
            # num_char_layers * self.size, .c and .h for each, backward and forward
            full_size = 4 * config.num_char_layers * self.size
            new_shape = tf.pack([batch_size, batch_verse_len, full_size])
            both_states_expanded = tf.reshape(both_states, new_shape)

        if is_training and config.keep_prob < 1:
            both_states_expanded = tf.nn.dropout(both_states_expanded, config.keep_prob)

        verse_lengths = tf.reshape(verse_lengths, [-1])
        outputs, last_states = tf.nn.dynamic_rnn(cell=word_level_cell,
                                                 dtype=data_type(),
                                                 sequence_length=verse_lengths,
                                                 inputs=both_states_expanded)
        self._final_state = last_states[-1]
        self._outputs = outputs

    @property
    def input(self):
        return self._input

    @property
    def initial_states(self):
        return self._initial_states

    @property
    def final_state(self):
        return self._final_state

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_size(self):
        return 2*self.size

class ConcatLearn(object):
    def __init__(self, is_training, config, labels, rnn_paths, verse_lengths, feed_forward_path, device="/cpu:0"):
        self.rnn_paths = rnn_paths
        self.feed_forward_path = feed_forward_path

        self._final_state = [m.final_state for m in self.rnn_paths]

        verse_len = tf.shape(self.rnn_paths[0].outputs)[1]
        # separate batch dimension into lists
        # rnn_paths * [batch_size, verse_len, output_size]
        outputs = [tf.reshape(path.outputs,[-1,path.output_size]) for path in self.rnn_paths]
        # rnn_paths * [batch_size * verse_len, output_size]
        outputs = tf.concat(1, outputs)
        # (batch_size * verse_len) * [sum(output_sizes)]

            # concat_final_shape = tf.shape(combined_outputs[0])[1]
            # concat_verse_shape = tf.shape(combined_outputs[0])[0]
            # outputs_flat = tf.reshape(combined_outputs, tf.pack([-1, concat_verse_shape * concat_final_shape]))
            # # batch_size * [verse_len * sum(output_sizes)])

        # [batch_size, ffp.output_size]
        feed_forward_tiled = tf.tile(self.feed_forward_path.output, tf.pack([verse_len, 1]))
        # (batch_size * verse_len) *[ffp.output_size]
        outputs_flat = tf.concat(1, [feed_forward_tiled, outputs])
        # batch_size * [feed_forward * verse_len * sum(output_sizes)])
        final_unit_size = sum([path.output_size for path in self.rnn_paths]) + self.feed_forward_path.output_size



        # final softmax layer

        # Output layer weights
        softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=data_type())
        print "vocab size:", config.vocab_size
        softmax_W = tf.get_variable(
            name="softmax_w",
            initializer=tf.random_normal_initializer(),
            shape=[final_unit_size, config.vocab_size],
            dtype=data_type())

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        logits_flat = tf.batch_matmul(outputs_flat, softmax_W) + softmax_b

        # Calculate the losses
        y_flat =  tf.reshape(labels, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses

        # Bring back to [B, T] shape
        masked_losses = tf.reshape(masked_losses,  tf.shape(labels))

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(verse_lengths, data_type())
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        self._cost = cost = mean_loss

        if not is_training:
            self.probs_flat = tf.nn.softmax(logits_flat)
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
    def initial_state(self):
        return [s for path in self.rnn_paths for s in path._initial_states]
    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op


class FeedForwardPath(object):
    def __init__(self, is_training, input_data, input_size, config):
        self.is_training = is_training
        self.keep_prob = config.keep_prob
        self.sizes = config.feed_forward_sizes
        self.input_size = input_size
        input_sizes = [self.input_size] + self.sizes[:-1]
        layers = []
        layer_input = tf.cast(input_data, data_type())
        for i, sizes in enumerate(zip(self.sizes, input_sizes)):
            sz, input_sz = sizes
            name = "DenseLayer{}".format(i)
            layer_input = self.layer(layer_input, input_sz, sz, name=name)
            layers.append(layer_input)
        self.output = layer_input

    def layer(self, input_data, input_size, size, name='layer'):
        with tf.variable_scope(name):
            W = self.weight_variable([input_size, size])
            b = self.bias_variable([size])
            layer = tf.nn.relu(tf.matmul(input_data, W) + b)
            if self.is_training and self.keep_prob < 1:
                layer = tf.nn.dropout(layer, self.keep_prob)
        return layer

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        W = tf.get_variable(
            name="W",
            initializer=initial,
            dtype=data_type())
        return W

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        b = tf.get_variable('b', initializer=initial, dtype=data_type())
        return b

    @property
    def output_size(self):
        return self.sizes[-1]


class FullLNModel(object):
    def __init__(self, is_training, config, input_, device="/cpu:0"):
        self.input_ = input_
        rapper0 = input_.rapper0
        verse_length = input_.verse_length
        word_length = input_.word_length[0]
        labels = input_.labels
        feat_names = ['chars', 'phones', 'stresses']
        feats = [input_[f] for f in feat_names]
        seq_lengths =[input_['chars_lengths'], input_['phones_lengths'], input_['stresses_lengths']]
        rnn_paths = []
        for i, feat in enumerate(feats):
            lengths = seq_lengths[i]
            rnn_input = RNNInput(feature=feat,
                                 word_lengths=lengths,
                                 sequence_lengths=verse_length,
                                 verse_length=verse_length,
                                 word_length=word_length)
            with tf.variable_scope("RNNPath_{}".format(feat_names[i])):
                rnn_path = RNNPath(is_training=is_training,
                                   input_=rnn_input,
                                   config=config)
                rnn_paths.append(rnn_path)

        with tf.variable_scope("FeedForwardPath"):
            feed_forward_path = FeedForwardPath(is_training,
                                                input_data=rapper0,
                                                input_size=config.rap_vec_size,
                                                config=config)

        with tf.variable_scope("ConcatLearn"):
            self.concat_learn = ConcatLearn(is_training=is_training,
                                       labels=labels,
                                       rnn_paths=rnn_paths,
                                       feed_forward_path=feed_forward_path,
                                       verse_lengths=verse_length,
                                       config=config)

    def assign_lr(self, session, lr_value):
        self.concat_learn.assign_lr(session, lr_value)

    @property
    def cost(self):
        return self.concat_learn.cost

    @property
    def input(self):
        return self.input_

    @property
    def initial_state(self):
        return self.concat_learn.initial_state

    @property
    def final_state(self):
        return self.concat_learn.final_state

    @property
    def lr(self):
        return self.concat_learn.lr
    @property
    def train_op(self):
        return self.concat_learn.train_op

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_word_layers = 2
    num_char_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    feed_forward_sizes = [50, 20, 10]
    def __init__(self, vocab_size, char_vocab_size, rap_vec_size):
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_word_layers = 2
    num_char_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    feed_forward_sizes = [100, 50, 25]
    def __init__(self, vocab_size, char_vocab_size, rap_vec_size):
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_word_layers = 2
    num_char_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    feed_forward_sizes = [200, 150, 50]
    def __init__(self, vocab_size, char_vocab_size, rap_vec_size):
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_word_layers = 2
    num_char_layers = 2
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 4
    feed_forward_sizes = [4, 3, 2]

    def __init__(self, vocab_size, char_vocab_size, rap_vec_size):
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size

def get_config(vocab_size, char_vocab_size, rap_vec_size):
    if FLAGS.model == "small":
        return SmallConfig(vocab_size, char_vocab_size, rap_vec_size)
    elif FLAGS.model == "medium":
        return MediumConfig(vocab_size, char_vocab_size, rap_vec_size)
    elif FLAGS.model == "large":
        return LargeConfig(vocab_size, char_vocab_size, rap_vec_size)
    elif FLAGS.model == "test":
        return TestConfig(vocab_size, char_vocab_size, rap_vec_size)
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def get_train_file():
    return FLAGS.train_filename
def get_valid_file():
    return FLAGS.train_filename


def run_epoch(session, model, eval_op=None, verbose=False):
    # TODO: initial_state and final_state both need to include both char-level and word-level RNN states
    # TODO: feed_dict feeds the previous state to the next step every time

    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, s in enumerate(model.initial_state):
            print "State:", s
            for j, (c, h) in enumerate(s):
                print "C:", c
                print "H:", h
                print "State[i]:", state[i]
                print "State[i][j]:", state[i][j]
                print "State[i][j].c:", state[i][j].c
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        print "feed_dict:"
        print feed_dict
        vals = session.run(fetches, feed_dict)
        print "vals:"
        print vals
        cost = vals["cost"]
        state = vals["final_state"]
        print vals.keys()
        # TODO: need to get num_steps from verse_length after running
        # access the verse_length variable from the model, and include it
        # in fetches with a key I specify

        costs += cost
        # iters += model.input.num_steps

        # if verbose and step % (model.input.epoch_size // 10) == 10:
            # print("%.3f perplexity: %.3f speed: %.0f wps" %
                  # (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   # iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def main(_):
    extractor = RapFeatureExtractor(train_filenames=[],
                                    valid_filenames=[],
                                    from_config=True,
                                    config_file='data/config_test.p')

    vocab_size = extractor.vocab_length
    char_vocab_size = extractor.char_vocab_length
    rap_vec_size = extractor.len_rapper_vector
    train_filename = get_train_file()
    valid_filename = get_valid_file()
    config = get_config(vocab_size, char_vocab_size, rap_vec_size)
    eval_config = get_config(vocab_size, char_vocab_size, rap_vec_size)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.name_scope("Train"):
            train_ln_input = LNInput(extractor=extractor, config=config, filename=train_filename, name="TrainInput")
            print "initializing training model:"
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = FullLNModel(is_training=True, config=config, input_=train_ln_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)
        with tf.name_scope("Valid"):
            print "initializing valid model:"
            valid_ln_input = LNInput(extractor, TestConfig, filename=valid_filename, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = FullLNModel(is_training=False, config=config, input_=valid_ln_input)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        print "initializing session:"
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            # test_perplexity = run_epoch(session, mtest)
            # print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


        # # Start populating the filename queue.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # batches = []
        # for i in range(10):
            # # Retrieve a single instance:
            # batched_parsed = sess.run(ln_input.input_data)
            # batches.append(batched_parsed)
        # coord.request_stop()
        # coord.join(threads)

if __name__ == "__main__":
  tf.app.run()
