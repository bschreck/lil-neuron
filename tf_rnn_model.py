import tensorflow as tf
import numpy as np
import time
import random
import copy

import tf_reader as reader
from extract_features import RapFeatureExtractor

import pdb

# TODO: change hidden size of each RNN path separately
# Use ADAM optimizer
# Add in pronunciation of rappers
# Start with GlovE or Word2Vec vectors
# Learn word vectors using bidirectional rnn with a single layer of word-level
# Freeze these vectors and train using 3 levels of word-level


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("train_filename", 'data/tf_train_data.txt',
                    "where the training data is stored.")
flags.DEFINE_string("valid_filename", 'data/tf_valid_data.txt',
                    "where the validation data is stored.")
flags.DEFINE_string("test_filename", 'data/tf_test_data.txt',
                    "where the test data is stored.")
flags.DEFINE_string("extractor_config_file", 'data/config.p',
                    "Config info for RapFeatureExtractor")
flags.DEFINE_string("save_path", 'models',
                    "Model output directory.")
flags.DEFINE_string("device", '/gpu:0',
                    "Preferred device.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("generate", False,
                  "If True, generate text instead of training")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class LNInput(object):
    """The input data."""

    def __init__(self, extractor, config, filename, name=None):
        self.extractor = extractor
        self.batch_size = config.batch_size
        self.max_num_steps = config.max_num_steps
        self.filename = filename
        self._epoch_size = reader.num_batches(extractor, self.batch_size, self.max_num_steps, self.filename)
        print "epoch size:", self._epoch_size
        self.input_data, _, _ = reader.batched_data_producer(self.extractor, self.batch_size, self.max_num_steps, self.filename, name=name)
        self.verse_length = self.input_data["labels"].get_shape()[:1]

    @property
    def epoch_size(self):
        return self._epoch_size

    def __getattr__(self, key):
        return self.input_data[key]

    def __getitem__(self, key):
        return self.input_data[key]

class GeneratorInput(object):
    def __init__(self, extractor, config, input_data, name=None):
        self.extractor = extractor
        self.batch_size = config.batch_size
        self.input_data = input_data
        self.verse_length = input_data["labels"].get_shape()[:1]
        self._epoch_size = 1

    @property
    def input_tensors(self):
        return self.input_data.values()

    @property
    def epoch_size(self):
        return self._epoch_size

    def __getattr__(self, key):
        return self.input_data[key]

    def __getitem__(self, key):
        return self.input_data[key]


class RNNInput(object):
    def __init__(self, feature, word_lengths, verse_lengths):
        self.feature = feature
        self.word_lengths = word_lengths
        self.verse_lengths = verse_lengths


class RNNPath(object):
    def __init__(self, is_training, config, input_, device=FLAGS.device):
        self.device = device
        self._input = input_
        feature = input_.feature
        batch_size = config.batch_size
        char_vocab_size= config.char_vocab_size
        word_lengths = input_.word_lengths
        verse_lengths = input_.verse_lengths
        feature_shape = tf.shape(feature)
        batch_verse_len = feature_shape[1]

        self.size = config.hidden_size

        with tf.device(device):
            # Can experiment with different settings
            lstm_args = dict(use_peepholes=False,
                             cell_clip=None,
                             initializer=None,
                             num_proj=None,
                             proj_clip=None,
                             num_unit_shards=1,
                             num_proj_shards=1,
                             forget_bias=1.0,
                             state_is_tuple=True)
            char_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.size, **lstm_args)
            word_lstm_cell = tf.nn.rnn_cell.LSTMCell(2 * self.size, **lstm_args)

            if is_training and config.keep_prob < 1:
                char_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(char_lstm_cell,
                                                               output_keep_prob=config.keep_prob)
                word_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(word_lstm_cell,
                                                               output_keep_prob=config.keep_prob)
            word_level_cell = tf.nn.rnn_cell.MultiRNNCell([word_lstm_cell] * config.num_word_layers,
                                                          state_is_tuple=True)
            char_level_cell_fw = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell] * config.num_char_layers,
                                                             state_is_tuple=True)
            char_level_cell_bw = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell] * config.num_char_layers,
                                                             state_is_tuple=True)

            self._initial_states = [word_level_cell.zero_state(batch_size, data_type()),
                                    char_level_cell_fw.zero_state(batch_size * batch_verse_len, data_type()),
                                    char_level_cell_bw.zero_state(batch_size * batch_verse_len, data_type())]

            # [batch_size, verse_len, word_len]
            new_shape = tf.pack([batch_size * batch_verse_len, -1])
            input_data_concat = tf.reshape(feature, new_shape)
            # [batch_size * verse_len, word_len]

            embedding = tf.get_variable(
                "embedding", [char_vocab_size, self.size], dtype=data_type())
            embed_inputs = tf.nn.embedding_lookup(embedding, input_data_concat)
            # [batch_size * verse_len, word_len, size]

            word_lengths = tf.reshape(word_lengths, tf.pack([batch_size * batch_verse_len]))

            _, bistates = tf.nn.bidirectional_dynamic_rnn(char_level_cell_fw,
                                                          char_level_cell_bw,
                                                          embed_inputs,
                                                          sequence_length=word_lengths,
                                                          dtype=data_type(),
                                                          swap_memory=True,
                                                          time_major=False,
                                                          scope=None)
            fw_states, bw_states = bistates
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
            verse_lengths = tf.tile(verse_lengths, [batch_size])
            outputs, last_states = tf.nn.dynamic_rnn(cell=word_level_cell,
                                                     dtype=data_type(),
                                                     sequence_length=verse_lengths,
                                                     inputs=both_states_expanded)
            self._final_states = [last_states, fw_states, bw_states]
            self._outputs = outputs

    @property
    def input(self):
        return self._input

    @property
    def initial_states(self):
        return self._initial_states

    @property
    def final_states(self):
        return self._final_states

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_size(self):
        return 2 * self.size


class ConcatLearn(object):
    def __init__(self, is_training, config, labels, rnn_paths, verse_lengths,
                 context_network, device=FLAGS.device):
        self.rnn_paths = rnn_paths
        self.context_network = context_network

        with tf.device(device):
            verse_len = tf.shape(self.rnn_paths[0].outputs)[1]
            # separate batch dimension into lists
            # rnn_paths * [batch_size, verse_len, output_size]
            outputs = [tf.reshape(path.outputs, [-1, path.output_size]) for path in self.rnn_paths]
            context_output = tf.reshape(self.context_network.outputs, [-1, self.context_network.output_size])
            outputs.append(context_output)
            # rnn_paths * [batch_size * verse_len, output_size]
            outputs_flat = tf.concat(1, outputs)
            # (batch_size * verse_len) * [sum(output_sizes)]

            final_unit_size = sum([path.output_size for path in self.rnn_paths]) + self.context_network.output_size
            # print "final_unit_size:", final_unit_size
            # # one more dense layer to shrink size
            # final_dense_W = tf.get_variable("final_dense",
                # initializer=tf.random_normal_initializer(),
                # shape=[final_unit_size, config.final_dense_size],
                # dtype=data_type())
            # final_dense_b = tf.get_variable("final_dense_b", [config.final_dense_size], dtype=data_type())

            # final_dense_out = tf.batch_matmul(outputs_flat, final_dense_W) + final_dense_b
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
            y_flat = tf.reshape(labels, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

            # Mask the losses
            mask = tf.sign(tf.to_float(y_flat))
            masked_losses = mask * losses

            # Bring back to [B, T] shape
            masked_losses = tf.reshape(masked_losses, tf.shape(labels))

            # Calculate mean loss
            mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(verse_lengths, data_type())
            mean_loss = tf.reduce_mean(mean_loss_by_example)
            self._cost = cost = mean_loss

            self._probs_flat = None
            if not is_training:
                self._probs_flat = tf.nn.softmax(logits_flat)
                return

            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            aggmeth = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,
                                                           aggregation_method=aggmeth),
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
    def output_probs(self):
        return self._probs_flat

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return [s for path in self.rnn_paths for s in path.initial_states] + [self.context_network.initial_states]

    @property
    def final_state(self):
        return [s for path in self.rnn_paths for s in path.final_states] + [self.context_network.final_states]

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class ContextNetwork(object):
    def __init__(self, is_training, input_data, input_size, verse_lengths,
                 config, device=FLAGS.device):
        self.is_training = is_training
        self.input_data= input_data
        self.paths = []
        self.verse_lengths = verse_lengths
        self.batch_size = config.batch_size
        self.keep_prob = config.keep_prob
        self.size = config.hidden_context_size

        all_rappers = tf.concat(2, input_data)

        name = "DenseLayer"
        input_shape = tf.shape(input_data[0])
        tensor_verse_len = input_shape[1]
        new_shape = tf.pack([self.batch_size * tensor_verse_len, len(input_data) * input_size])
        layer_input = tf.reshape(all_rappers, new_shape, name="PreDenseReshape")
        layer_input = tf.cast(layer_input, data_type())
        layer_output = self.layer(layer_input, len(input_data) * input_size, self.size, name=name,
                                 device=device)
        new_shape = tf.pack([self.batch_size, tensor_verse_len, self.size])
        rnn_input = tf.reshape(layer_output, new_shape, name="PostDenseReshape")

        self.size = config.hidden_context_size

        with tf.device(device):
            # Can experiment with different settings
            lstm_args = dict(use_peepholes=False,
                             cell_clip=None,
                             initializer=None,
                             num_proj=None,
                             proj_clip=None,
                             num_unit_shards=1,
                             num_proj_shards=1,
                             forget_bias=1.0,
                             state_is_tuple=True)
            cell = tf.nn.rnn_cell.LSTMCell(self.size, **lstm_args)

            if is_training and config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=config.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_context_layers, state_is_tuple=True)
            self._initial_states = cell.zero_state(self.batch_size, data_type())


            verse_lengths = tf.reshape(verse_lengths, [-1])
            verse_lengths = tf.tile(verse_lengths, [self.batch_size])
            outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                     dtype=data_type(),
                                                     sequence_length=verse_lengths,
                                                     inputs=rnn_input)
        self._final_states = last_states
        self._outputs = outputs

    @property
    def initial_states(self):
        return self._initial_states

    @property
    def final_states(self):
        return self._final_states

    @property
    def outputs(self):
        return self._outputs

    @property
    def num_paths(self):
        return len(self.input_data)

    @property
    def output_size(self):
        return self.size

    def layer(self, input_data, input_size, size, name='layer',
              device=FLAGS.device):
        with tf.variable_scope(name), tf.device(device):
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


class FeedForwardPath(object):
    def __init__(self, is_training, input_data, input_size, config,
                 device=FLAGS.device):
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
            layer_input = self.layer(layer_input, input_sz, sz, name=name,
                                     device=device)
            layers.append(layer_input)
        self.output = layer_input

    def layer(self, input_data, input_size, size, name='layer',
              device=FLAGS.device):
        with tf.variable_scope(name), tf.device(device):
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
    def __init__(self, is_training, config, input_, device=FLAGS.device):
        self.input_ = input_
        self.config = config
        num_rappers = config.num_rappers
        rappers = []
        for r in xrange(num_rappers):
            rappers.append(input_["rapper" + str(r)])

        labels = input_.labels
        feat_names = ['chars', 'phones', 'stresses']
        feats = [input_[f] for f in feat_names]
        self.seq_lengths = [input_['chars.lengths'], input_['phones.lengths'], input_['stresses.lengths']]
        rnn_paths = []
        for i, feat in enumerate(feats):
            lengths = self.seq_lengths[i]
            rnn_input = RNNInput(feature=feat,
                                 word_lengths=lengths,
                                 verse_lengths=self.verse_length)
            with tf.variable_scope("RNNPath_{}".format(feat_names[i])):
                rnn_path = RNNPath(is_training=is_training,
                                   input_=rnn_input,
                                   config=config,
                                   device=device)
                rnn_paths.append(rnn_path)

        with tf.variable_scope("ContextNetwork"):
            context_network = ContextNetwork(is_training,
                                             input_data=rappers,
                                             input_size=config.rap_vec_size,
                                             verse_lengths=self.verse_length,
                                             config=config,
                                             device=device)

        with tf.variable_scope("ConcatLearn"):
            self.concat_learn = ConcatLearn(is_training=is_training,
                                            labels=labels,
                                            rnn_paths=rnn_paths,
                                            context_network=context_network,
                                            verse_lengths=self.verse_length,
                                            config=config,
                                            device=device)

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def verse_length(self):
        return self.input_.verse_length

    @property
    def input_tensors(self):
        return self.input_.input_tensors

    def assign_lr(self, session, lr_value):
        self.concat_learn.assign_lr(session, lr_value)

    @property
    def output_probs(self):
        return self.concat_learn.output_probs

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


class Config(object):
    def __init__(self, vocab_size, char_vocab_size, rap_vec_size, num_rappers):
        self.char_vocab_size = char_vocab_size
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size
        self.num_rappers = num_rappers

    def to_eval_gen_config(self):
        config = copy.copy(self)
        config.batch_size = 1
        return config


class SmallConfig(Config):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_word_layers = 2
    num_char_layers = 1
    num_context_layers = 1
    max_num_steps = 20
    hidden_size = 200
    hidden_context_size = 25
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    feed_forward_sizes = [50, 20, 10]
    feed_forward_final_size = 10
    final_dense_size = 20


class MediumConfig(Config):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_word_layers = 2
    num_char_layers = 1
    num_context_layers = 1
    max_num_steps = 35
    hidden_size = 650
    hidden_context_size = 50
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    feed_forward_sizes = [100, 50, 25]
    feed_forward_final_size = 25
    final_dense_size = 30


class LargeConfig(Config):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_word_layers = 2
    num_char_layers = 1
    num_context_layers = 2
    max_num_steps = 35
    hidden_size = 1500
    hidden_context_size = 100
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    feed_forward_sizes = [200, 150, 50]
    feed_forward_final_size = 50
    final_dense_size = 50


class TestConfig(Config):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_word_layers = 2
    num_char_layers = 2
    num_context_layers = 1
    max_num_steps = 2
    hidden_size = 2
    hidden_context_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 4
    feed_forward_sizes = [4, 3, 2]
    feed_forward_final_size = 2
    final_dense_size = 2

def get_config(*args):
    if FLAGS.model == "small":
        return SmallConfig(*args)
    elif FLAGS.model == "medium":
        return MediumConfig(*args)
    elif FLAGS.model == "large":
        return LargeConfig(*args)
    elif FLAGS.model == "test":
        return TestConfig(*args)
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def get_train_file():
    return FLAGS.train_filename


def get_valid_file():
    return FLAGS.train_filename


def get_test_file():
    return FLAGS.test_filename


def run_epoch(session, model, eval_op=None, verbose=False):
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
            for j, (c, h) in enumerate(s):
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        verse_length = model.verse_length.as_list()[0]

        costs += cost
        iters += verse_length * model.batch_size

        every10 = model.input.epoch_size // 10
        print_output = (every10 == 0 or step % every10 == 10)
        if verbose and print_output:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / verse_length),
                   iters / (time.time() - start_time)))

    return np.exp(costs / verse_length)

def generate_text(extractor, gen_config, rappers, starter):
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-gen_config.init_scale,
                                                    gen_config.init_scale)
        with tf.name_scope("Train"):
            lines = ["<eos>\n", "<eov>\n", "<eor>\n", ""]
            for rapper in rappers:
                lines[-1] += "(NRP: {})".format(rapper)
            starter = starter.replace(". ", "<eos> ")
            lines.append(starter)
            tensor_dict, input_data, init_op_local = extractor.gen_features_from_starter(rappers, lines)

            gen_input = GeneratorInput(extractor=extractor, config=gen_config,
                                       input_data=tensor_dict, name="GeneratorInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = FullLNModel(is_training=False, config=gen_config,
                                input_=gen_input)
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=20)
        text = ""
        with sv.managed_session(config=tf_config) as session:
            # Restore variables from disk.
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                sv.saver.restore(session, ckpt.model_checkpoint_path)
                print "Model restored from file " + FLAGS.save_path
            else:
                raise Exception("Model ckpt not found")

            state = session.run(m.initial_state)

            get_word = extractor.get_word_from_int

            end_of_rap = False
            new_context = -1
            first_time = True
            while not end_of_rap:
                feed_dict = {}
                for i, s in enumerate(m.initial_state):
                    for j, (c, h) in enumerate(s):
                        feed_dict[c] = state[i][j].c
                        feed_dict[h] = state[i][j].h

                for k, t in tensor_dict.iteritems():
                    feed_dict[t] = input_data[k]
                output_probs, state = session.run([m.output_probs, m.final_state],
                                                  feed_dict)

                x = sample(output_probs[-1], 0.9)
                if new_context > -1:
                    rap_vectors = extractor.update_rap_vectors(x, new_context)
                    input_data.update(rap_vectors)

                word = get_word(x)
                if word == "<eos>":
                    text += "\n"
                    new_context = -1
                elif word == "<eov>":
                    text += "\n\n"
                elif word == "<eor>":
                    end_of_rap = True
                elif word == "<nrp>":
                    new_context += 1
                    if new_context > extractor.max_nrps:
                        new_context = -1
                else:
                    text += " " + word
                print word.encode('utf-8')
                feats = extractor.update_features(x)
                input_data.update(feats)
                if first_time:
                    # only include first context vector
                    # since we now only have verses of length 1
                    for k, v in input_data.iteritems():
                        if k.startswith("rapper"):
                            first_vec = v[:, 0, :]
                            first_vec = first_vec[np.newaxis, :]
                            input_data[k] = first_vec
                first_time = False
    print text.encode('utf-8')
    return


def sample(a, temperature=1.0):
    # necessary because TF softmax sometimes
    # produces probabilities that sum to greater than 1
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    r = random.random() # range: [0,1)
    total = 0.0
    for i in range(len(a)):
        total += a[i]
        if total > r:
            return i
    return len(a) - 1


def main(argv):

    if len(argv) > 1:
        FLAGS.model = argv[1]

    if len(argv) > 2 and argv[2].startswith('g'):
        FLAGS.generate = True
    else:
        FLAGS.generate = False

    extractor = RapFeatureExtractor(train_filenames=[],
                                    valid_filenames=[],
                                    from_config=True,
                                    config_file=FLAGS.extractor_config_file)
    vocab_size = extractor.vocab_length + 1
    char_vocab_size = extractor.char_vocab_length + 1
    rap_vec_size = extractor.len_rapper_vector
    num_rappers = extractor.max_nrps
    train_filename = get_train_file()
    valid_filename = get_valid_file()
    test_filename = get_test_file()
    config = get_config(vocab_size, char_vocab_size, rap_vec_size, num_rappers)
    eval_gen_config = config.to_eval_gen_config()

    if FLAGS.generate:
        rappers = ["Tyler, The Creator"]
        starter = "I eat people"
        generate_text(extractor, eval_gen_config, rappers, starter)
        return

    print "initializing session:"
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
            valid_ln_input = LNInput(extractor=extractor, config=config, filename=valid_filename, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = FullLNModel(is_training=False, config=config, input_=valid_ln_input)
            tf.scalar_summary("Validation Loss", mvalid.cost)
        with tf.name_scope("Test"):
            print "initializing test model:"
            test_ln_input = LNInput(extractor=extractor, config=eval_gen_config, filename=test_filename, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = FullLNModel(is_training=False, config=eval_gen_config,
                                    input_=test_ln_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=20)
        with sv.managed_session(config=tf_config) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
