import tensorflow as tf
import numpy as np
import time
import random
import copy
import sys

import tf_reader as reader
from extract_features import RapFeatureExtractor
from generate_lyric_files import format_rapper_name
from supervisor import PartialSupervisor, PartialSessionManager

import pdb

# TODO: change hidden size of each RNN path separately
# Use ADAM optimizer
# Add in pronunciation of rappers
# Start with GlovE or Word2Vec vectors
# Learn word vectors using bidirectional rnn with a single layer of word-level
# Freeze these vectors and train using 3 levels of word-level

# HOW TO START WITH WORD VECTORS
# get full google news corpus (or other corpus, look into what GloVe is trained on)
# add in my rap corpus, which includes slang words
# retrain to produce vectors for all words in my corpus
# ACTUALLY check on how many of my rap words are in the pretrained GloVe vectors (most of them probably are)
# for how to do word embeddings: http://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
# freeze weights inititally for a few epochs until network is decent, then start training them

# Use standard embedding (not bidirectional rnn) for word vectors, using double layer of word-level
# Freeze these and then use triple level
# Try both bidirectional and standard embedding for pronunciations, and experiment with freezing them as well

# Try to generate embeddings for both pronunciations and words simultaneiously, then freeze them both
# Either remove stresses, add them back in to the original pronunciations,
# or encode each pronuncation as a tuple of (phone, stress) into the same RNN
# make sure this happens from extract_features

# NRP symbols should be taken out of sequence features and just put in context, with both rap vectors as well as rapper symbols
# labels will then be both the next word and the next set of rappers (max 3 of them, with a 0 symbol for no rapper)
# can possibly include verse/chorus/hook symbols there too but don't worry about them for now
# Use mean vector for unknown words, and learn them first


# INCORPORATING TOPIC INFO AND CONTEXT FEATURES:
# https://pdfs.semanticscholar.org/04e0/fefb859f4b02b017818915a2645427bfbdb2.pdf
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: test, small, medium, large.")
flags.DEFINE_string("train_filename", 'data/tf_train_data_full.txt',
                    "where the training data is stored.")
flags.DEFINE_string("valid_filename", 'data/tf_valid_data_full.txt',
                    "where the validation data is stored.")
flags.DEFINE_string("test_filename", 'data/tf_test_data_full.txt',
                    "where the test data is stored.")
flags.DEFINE_string("extractor_config_file", 'data/config_full.p',
                    "Config info for RapFeatureExtractor")
flags.DEFINE_string("word_vector_file", 'data/word_vectors/glove_retro.txt',
                    "Config info for RapFeatureExtractor")

flags.DEFINE_string("processed_word_vector_file", 'data/processed_word_vector_array.p',
                    "")
flags.DEFINE_string("processed_pron_vector_file", 'data/processed_pron_vector_array.p',
                    "")
flags.DEFINE_string("save_path", 'models',
                    "Model output directory.")
flags.DEFINE_string("device", '/gpu:0',
                    "Preferred device.")
flags.DEFINE_string("alternate_device", '/gpu:1',
                    "Preferred device.")
flags.DEFINE_string("cpu", '/cpu:0',
                    "Preferred device.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("generate", False,
                  "If True, generate text instead of training")
flags.DEFINE_bool("train_word_vectors", False,
                  "whether to train word embeddings")
flags.DEFINE_bool("train_pron_embedding", False,
                  "whether to train pronunciation embeddings")
flags.DEFINE_bool("train_context_embedding", False,
                  "whether to train context (i.e. rap vectors) embeddings")
flags.DEFINE_bool("restore_from_checkpoint", False,
                  "whether to force training to restore from a checkpoint")
flags.DEFINE_integer("phase", 2,
                  "")

flags.DEFINE_integer("num_embed_shards", 8, "")

flags.DEFINE_string("find_epoch_size", False, "")
flags.DEFINE_integer("train_epoch_size", 30328, "")
flags.DEFINE_integer("valid_epoch_size", 30328, "")
flags.DEFINE_integer("test_epoch_size", 1387630, "")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def partitioner():
    max_shard_bytes = (24 << 20) - 1
    return tf.variable_axis_size_partitioner(max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None)

def initializer():
    return tf.contrib.layers.variance_scaling_initializer(dtype=data_type())


class LNInput(object):
    """The input data."""

    def __init__(self, extractor, config, filename, epoch_size=None, name=None):
        self.extractor = extractor
        self.batch_size = config.batch_size
        self.max_num_steps = config.max_num_steps
        self.filename = filename
        if FLAGS.find_epoch_size:
            print self.batch_size, self.max_num_steps
            self._epoch_size = reader.num_batches(extractor, self.batch_size, self.max_num_steps, self.filename)
            batches = reader.run_and_return_batches(extractor, 1, self.batch_size, self.max_num_steps, self.filename)
            pdb.set_trace()
        else:
            self._epoch_size = epoch_size
        print "epoch size:", self._epoch_size
        self.input_data, _, _ = reader.batched_data_producer(self.extractor, self.batch_size, self.max_num_steps, self.filename, name=name)

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
    def __init__(self, words, context, context_size):
        self.words = words
        self.context = context
        self.context_size = context_size


class RNNPath(object):
    def __init__(self, is_training, config, input_,
                 train_word_vectors=FLAGS.train_word_vectors,
                 train_pron_embedding=FLAGS.train_pron_embedding,
                 train_context_embedding=FLAGS.train_context_embedding,
                 device=FLAGS.device):
        if not is_training:
            train_word_vectors = False
            train_pron_embedding = False
            train_context_embedding = False
        self.device = device
        self._input = input_
        words = input_.words

        self.context = input_.context
        self.context_size = input_.context_size

        self.batch_size = config.batch_size
        self.rnn_size = config.hidden_size
        self.keep_prob = config.keep_prob
        if FLAGS.phase == 1:
            self.num_layers = config.num_layers
        else:
            self.num_layers = config.phase_2_num_layers
        self.embedding_dim = config.embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.vocab_size = config.vocab_size
        self.max_num_steps = config.max_num_steps
        self.max_pron_length = config.max_pron_length

        with tf.device('/cpu:0'):
            # Can experiment with different settings
            lstm_args = dict(use_peepholes=False,
                             cell_clip=None,
                             initializer=initializer(),
                             num_proj=None,
                             proj_clip=None,
                             num_unit_shards=1,
                             num_proj_shards=1,
                             forget_bias=1.0,
                             state_is_tuple=True)
            cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, **lstm_args)

            if is_training and config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=config.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers,
                                               state_is_tuple=True)

            self._initial_state = cell.zero_state(self.batch_size, data_type())



        with tf.device('/cpu:0'):
            shard_size = int(self.vocab_size / FLAGS.num_embed_shards)
            shard_sizes = [shard_size for _ in xrange(FLAGS.num_embed_shards)]
            shard_sizes[-1] = self.vocab_size - (shard_size * (FLAGS.num_embed_shards - 1))

            self._embedding_placeholders = []
            self._embedding_inits = []
            embeddings = []
            for shard, shard_size in enumerate(shard_sizes):
                shard_embedding = tf.get_variable(
                        "word_vector_shard_{}".format(shard),
                        trainable=train_word_vectors,
                        shape=[shard_size, self.embedding_dim],
                        dtype=data_type())
                embeddings.append(shard_embedding)
                shard_embedding_placeholder = tf.placeholder(data_type(), [shard_size, self.embedding_dim])
                self._embedding_placeholders.append(shard_embedding_placeholder)

                self._embedding_inits.append(shard_embedding.assign(shard_embedding_placeholder))

            self._pron_lookup_placeholders = []
            self._pron_lookup_inits = []
            pron_lookups = []
            for shard, shard_size in enumerate(shard_sizes):
                shard_pron_lookup = tf.get_variable(
                        "pron_lookup_shard_{}".format(shard),
                        trainable=False,
                        shape=[shard_size, self.max_pron_length],
                        dtype=data_type())
                pron_lookups.append(shard_pron_lookup)
                shard_pron_lookup_placeholder = tf.placeholder(data_type(), [shard_size, self.max_pron_length])
                self._pron_lookup_placeholders.append(shard_pron_lookup_placeholder)

                self._pron_lookup_inits.append(shard_pron_lookup.assign(shard_pron_lookup_placeholder))



            pronunciation_embedding = tf.get_variable("pronunciation_embedding_W",
                    initializer=initializer(),
                    shape=[self.max_pron_length, self.embedding_dim],
                    trainable=train_pron_embedding,
                    dtype=data_type())
            pronunciation_b = tf.get_variable("pronunciation_embedding_b",
                                              initializer=initializer(),
                                              shape=[self.embedding_dim],
                                              trainable=train_context_embedding,
                                              dtype=data_type())

            context_embedding = tf.get_variable("context_embedding_W",
                    initializer=initializer(),
                    shape=[self.context_size, self.context_embedding_dim],
                    trainable=train_context_embedding,
                    dtype=data_type())
            context_b = tf.get_variable("context_embedding_b",
                                              initializer=initializer(),
                                              shape=[self.context_embedding_dim],
                                              trainable=train_context_embedding,
                                              dtype=data_type())

        with tf.device('/gpu:0'):
            embedding = tf.concat(0, embeddings)
            pron_lookup = tf.concat(0, pron_lookups)

            word_embed_inputs = tf.nn.embedding_lookup(embedding, words)
            pron_inputs = tf.nn.embedding_lookup(pron_lookup, words)
            flat_pron_inputs = tf.reshape(pron_inputs, [-1, self.max_pron_length])
            self.flat_pron_inputs = flat_pron_inputs
            pron_embed_inputs = tf.batch_matmul(flat_pron_inputs, pronunciation_embedding) + pronunciation_b
            flat_context_inputs = tf.reshape(self.context,
                                             [-1, self.context_size])
            context_embed_inputs = tf.batch_matmul(tf.cast(flat_context_inputs, data_type()), context_embedding) + context_b

            pron_embed_inputs = tf.reshape(pron_embed_inputs,
                                           [self.batch_size, -1, self.embedding_dim])
            context_embed_inputs = tf.reshape(context_embed_inputs,
                                              [self.batch_size, -1, self.context_embedding_dim])
            # TODO: try dropout, nonlinearity



            combined = tf.concat(2, [word_embed_inputs, pron_embed_inputs, context_embed_inputs])

            full_embedded_input_size = 2*self.embedding_dim + self.context_embedding_dim
        with tf.device('/cpu:0'):
            resize_layer_w = tf.get_variable("rnn_resize_w",
                                           initializer=initializer(),
                                           shape=[full_embedded_input_size, self.rnn_size],
                                           trainable=True,
                                           dtype=data_type())
            resize_layer_b = tf.get_variable("rnn_resize_b",
                                           initializer=initializer(),
                                           shape=[self.rnn_size],
                                           trainable=True,
                                           dtype=data_type())
        with tf.device('/gpu:0'):
            combined_flattened = tf.reshape(combined, [-1, full_embedded_input_size])
            rnn_input = tf.batch_matmul(combined_flattened, resize_layer_w) + resize_layer_b
            rnn_input = tf.reshape(rnn_input, [self.batch_size, -1, self.rnn_size])


            if is_training and self.keep_prob < 1:
                rnn_input = tf.nn.dropout(rnn_input, self.keep_prob)

        with tf.device('/gpu:0'):
            outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                     dtype=data_type(),
                                                     #sequence_length=verse_lengths,
                                                     inputs=rnn_input)
            self._final_state = last_states
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

    @property
    def output_size(self):
        return self.rnn_size
    @property
    def embedding_placeholders(self):
        return self._embedding_placeholders
    @property
    def embedding_inits(self):
        return self._embedding_inits
    @property
    def pron_lookup_placeholders(self):
        return self._pron_lookup_placeholders
    @property
    def pron_lookup_inits(self):
        return self._pron_lookup_inits


class Learn(object):
    def __init__(self, is_training, config, labels, input,
                 device=FLAGS.device):
        self.rnn_path = input
        self.context = input.context
        self.context_size = input.context_size
        self.vocab_size = config.vocab_size

        #with tf.device('/gpu:0'):
            #verse_len = tf.shape(self.rnn_path.outputs)[1]
        with tf.device('/gpu:0'):

            final_unit_size = self.rnn_path.output_size + self.context_size
	    print "final unit:", final_unit_size

            concat = tf.concat(2, [self.rnn_path.outputs, tf.cast(self.context, data_type())])
            outputs_flat = tf.reshape(concat, [-1, final_unit_size])

        with tf.device('/cpu:0'):

            softmax_b = tf.get_variable("softmax_b", [self.vocab_size],
					dtype=data_type())

            print "vocab size:", self.vocab_size
            softmax_W = tf.get_variable(
                name="softmax_w",
                initializer=initializer(),
                shape=[final_unit_size, self.vocab_size],
                partitioner=partitioner(),
                dtype=data_type())


        with tf.device('/gpu:0'):
            # Calculate logits and probs
            # Reshape so we can calculate them all at once
            print "OUTPUTS FLAT:", outputs_flat
            logits_flat = tf.matmul(outputs_flat, softmax_W.as_tensor()) + softmax_b
            self.logits_flat = logits_flat

            # Calculate the losses
            y_flat = tf.reshape(labels, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

            # Mask the losses
            mask = tf.sign(tf.cast(y_flat, data_type()))
            masked_losses = mask * losses

            # Bring back to [B, T] shape
            masked_losses = tf.reshape(masked_losses, tf.shape(labels))
            self.losses = masked_losses

            # Calculate mean loss
            mean_loss = tf.reduce_sum(masked_losses, reduction_indices=1)
            mean_loss = tf.reduce_mean(mean_loss)
            self._cost = cost = mean_loss

            self._probs_flat = None
            if not is_training:
                self._probs_flat = tf.nn.softmax(logits_flat)
                return

        with tf.device('/cpu:0'):
            self._lr = tf.Variable(0.0, trainable=False, dtype=data_type())
            tvars = tf.trainable_variables()
        with tf.device('/gpu:1'):
            aggmeth = tf.AggregationMethod.EXPERIMENTAL_TREE
            #aggmeth = tf.AggregationMethod.ADD_N
            tvars = tf.trainable_variables()
            aggmeth = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
            optimizer = tf.train.AdamOptimizer(self._lr)
            GATE_NONE = 0
            GATE_OP = 1
            GATE_GRAPH = 2

            gate_grad = GATE_NONE # GATE_OP, GATE_GRAPH
            grads_and_vars = optimizer.compute_gradients(cost, tvars, aggregation_method=aggmeth,
                                              gate_gradients=gate_grad)

            #clipped_grads, _ = tf.clip_by_global_norm([g for g,v in grads_and_vars],
            #                                  config.max_grad_norm)
            clipped_grads = [tf.clip_by_norm(g, config.max_grad_norm) for g,v in grads_and_vars]

        with tf.device('/cpu:0'):
            self._train_op = optimizer.apply_gradients(
                zip(clipped_grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

        with tf.device('/cpu:0'):
            self._new_lr = tf.placeholder(
                data_type(), shape=[], name="new_learning_rate")
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
        return self.rnn_path.initial_state

    @property
    def final_state(self):
        return self.rnn_path.final_state

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op



class FullLNModel(object):
    def __init__(self, is_training, config, input_, device=FLAGS.device):
        self.input_ = input_
        self.config = config
        num_rappers = config.num_rappers
        rappers = []

        for r in xrange(num_rappers):
            rappers.append(input_["rapper" + str(r)])
        context = tf.concat(2, rappers)

        labels = input_.labels

        self.rnn_input = RNNInput(words=input_.words,
                             context=context,
                             context_size = num_rappers * self.config.rap_vec_size)
        with tf.variable_scope("RNNPath"):
            self.rnn_path = RNNPath(is_training=is_training,
                                    input_=self.rnn_input,
                                    config=config,
                                    device=device)


        with tf.variable_scope("Learn"):
            self.learn = Learn(is_training=is_training,
                               labels=labels,
                               input=self.rnn_path,
                               config=config,
                               device=FLAGS.cpu)

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def input_tensors(self):
        return self.input_.input_tensors

    def assign_lr(self, session, lr_value):
        self.learn.assign_lr(session, lr_value)

    @property
    def output_probs(self):
        return self.learn.output_probs

    @property
    def cost(self):
        return self.learn.cost

    @property
    def input(self):
        return self.input_

    @property
    def initial_state(self):
        return self.rnn_path.initial_state

    @property
    def final_state(self):
        return self.rnn_path.final_state

    @property
    def lr(self):
        return self.learn.lr

    @property
    def train_op(self):
        return self.learn.train_op

    @property
    def embedding_placeholders(self):
        return self.rnn_path.embedding_placeholders
    @property
    def embedding_inits(self):
        return self.rnn_path.embedding_inits
    @property
    def pron_lookup_placeholders(self):
        return self.rnn_path.pron_lookup_placeholders
    @property
    def pron_lookup_inits(self):
        return self.rnn_path.pron_lookup_inits

    @property
    def context(self):
        return self.learn.context

class Config(object):
    def __init__(self, vocab_size, rap_vec_size, num_rappers, max_pron_length):
        self.vocab_size = vocab_size
        self.rap_vec_size = rap_vec_size
        self.num_rappers = num_rappers
        self.max_pron_length = max_pron_length
        self.original_batch_size = self.batch_size
        self.original_max_num_steps = self.max_num_steps

    def to_eval_gen_config(self):
        config = copy.copy(self)
        config.original_batch_size = config.batch_size
        config.original_max_num_steps = config.max_num_steps
        config.batch_size = 1
        config.max_num_steps = 1
        return config


class SmallConfig(Config):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    phase_2_num_layers = 3
    max_num_steps = 20
    hidden_size = 200
    embedding_dim = 300
    context_embedding_dim = 20
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


class MediumConfig(Config):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    phase_2_num_layers = 3
    max_num_steps = 30
    hidden_size = 650
    embedding_dim = 300
    context_embedding_dim = 30
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 10


class LargeConfig(Config):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    phase_2_num_layers = 3
    max_num_steps = 30
    hidden_size = 1500
    embedding_dim = 300
    context_embedding_dim = 50
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 15


class TestConfig(Config):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 2
    phase_2_num_layers = 3
    max_num_steps = 2
    hidden_size = 2
    embedding_dim = 300
    context_embedding_dim = 300
    max_epoch = 1
    max_max_epoch = 3
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 4

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

def shard_array(array, num_shards):
    shard_size = int(array.shape[0] / num_shards)
    sharded_arrays = []
    for i in xrange(num_shards):
        if i == num_shards - 1:
            shard = array[i * shard_size :]
        else:
            shard = array[i * shard_size : (i+1) * shard_size]
        sharded_arrays.append(shard)
    return sharded_arrays

def run_epoch(session, model, word_vectors=None, pronunciation_vectors=None, eval_op=None, verbose=False, extractor=None):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        #"logits": model.learn.logits_flat,
        #"losses": model.learn.losses,
        "final_state": model.final_state,
        "context": model.context,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    sharded_word_vectors = shard_array(word_vectors, FLAGS.num_embed_shards)
    sharded_pron_vectors = shard_array(pronunciation_vectors, FLAGS.num_embed_shards)

    embedding_feed_dict = {}
    embedding_init_vars = []
    if word_vectors is not None:
        embedding_init_vars.extend(model.embedding_inits)
        for shard, placeholder in enumerate(model.embedding_placeholders):
            embedding_feed_dict[placeholder] = sharded_word_vectors[shard]
    if pronunciation_vectors is not None:
        embedding_init_vars.extend(model.pron_lookup_inits)
        for shard, placeholder in enumerate(model.pron_lookup_placeholders):
            embedding_feed_dict[placeholder] = sharded_pron_vectors[shard]

    session.run(embedding_init_vars,
                feed_dict=embedding_feed_dict)


    feed_dict = {}
    for step in range(model.input.epoch_size):
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        verse_length = model.input.max_num_steps
        costs += cost
        iters += verse_length

        every10 = max(model.input.epoch_size // 100, 1)
        #print "wps: {}".format( iters * model.batch_size / (time.time() - start_time))
        print_output = (step == 0 or step % every10 == 0)
        if verbose and print_output:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
            sys.stdout.flush()
            print iters * model.batch_size

    return np.exp(costs / iters)

def generate_text(extractor, gen_config, rappers, starter):
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-gen_config.init_scale,
                                                    gen_config.init_scale)
        with tf.name_scope("Train"):
            words = ["<eos>", "<eov>", "<eor>"]
            rappers = [format_rapper_name(r) for r in rappers]
            words.append("<nrp:{}>".format(';'.join(rappers)))
            starter = starter.replace(". ", " <eos> ").replace('\n', ' ').split()
            words.extend(starter)
            tensor_dict, input_data, init_op_local = extractor.gen_features_from_starter(rappers, words)

            gen_input = GeneratorInput(extractor=extractor, config=gen_config,
                                       input_data=tensor_dict,
                                       name="GeneratorInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = FullLNModel(is_training=False, config=gen_config,
                                input_=gen_input)
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=20)
        text = ""
        with sv.managed_session(config=tf_config) as session:
            # new_saver = tf.train.import_meta_graph('my-model.meta')
            # new_saver.restore(sess, tf.train.latest_checkpoint('./'))
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
            first_time = True

            while not end_of_rap:
                feed_dict = {}
                for i, (c, h) in enumerate(m.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                for k, t in tensor_dict.iteritems():
                    feed_dict[t] = input_data[k]
                output_probs, state = session.run([m.output_probs, m.final_state],
                                                  feed_dict)

		print output_probs
                x = sample(output_probs[-1], 0.9)

                word = get_word(x)
                if word == "<eos>":
                    text += "\n"
                elif word == "<eov>":
                    text += "\n\n"
                elif word == "<eor>":
                    end_of_rap = True
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
    temperature = 1.0
    # necessary because TF softmax sometimes
    # produces probabilities that sum to greater than 1
    pdb.set_trace()
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    r = random.random() # range: [0,1)
    total = 0.0
    for i in range(len(a)):
        total += a[i]
        if total > r:
            print "first return", i
            return i
    print "second return"
    return len(a) - 1


def main(argv):

    if len(argv) > 1:
        FLAGS.model = argv[1]
        if FLAGS.model == "test":
            FLAGS.train_filename = "data/tf_data_tiny.txt"
            FLAGS.valid_filename = "data/tf_data_tiny.txt"
            FLAGS.test_filename  = "data/tf_data_tiny.txt"
            FLAGS.processed_word_vector_file = "data/processed_word_vector_array_tiny.p"
            FLAGS.processed_pron_vector_file = "data/processed_pron_vector_array_tiny.p"
            FLAGS.extractor_config_file = "data/config_tiny.p"
            FLAGS.train_epoch_size = 25
            FLAGS.valid_epoch_size = 25
            FLAGS.test_epoch_size = 200
            FLAGS.find_epoch_size = True
        FLAGS.save_path = "{}_models".format(FLAGS.model)
    if len(argv) > 2 and argv[2].startswith('g'):
        FLAGS.generate = True
    else:
        FLAGS.generate = False

    FLAGS.train_pron_embedding = False
    FLAGS.train_word_vectors = False
    FLAGS.train_context_embedding = False
    if FLAGS.phase > 1:
        FLAGS.train_pron_embedding = True
        FLAGS.train_word_vectors = True
        FLAGS.train_context_embedding = True
        FLAGS.restore_from_checkpoint = True


    # TODO: make sure all files are set to correct places
    # start with loading word vectors/prons from file set to False
    # and provide the filename of the glove vectors
    # figure out what I need to package and deliver to remote GPUs
    # (make sure I load pronunciations to a file first so I don't need mongo)
    # try to run on GPU, then keep updating slang words, and redo corpus with better rappers featured more
    # in future after learning a good vector embedding and pron embedding:
    #    learn multiple semantic models for different categories of thought (e.g. poetry, physics, machine learning)
    #    where each model is trained same way as here
    #    then when generating rap lyrics, at each time point
    #    sample from the different semantic models (a semantic model is the word embedding)
    #    while always using the rap pronunciation embedding
    print "initializing extractor"
    extractor = RapFeatureExtractor(from_config=True,
                                    config_file=FLAGS.extractor_config_file,
                                    pronunciation_vectors_file=FLAGS.processed_pron_vector_file,
                                    load_word_vectors_from_file=True,
                                    word_vectors_file=FLAGS.processed_word_vector_file,
                                    load_pronunciation_vectors_from_file=True)
    print "loading embeddings"
    pronunciation_vectors, max_pron_length = extractor.load_pronunciation_vectors()
    word_vectors = extractor.load_glove_vectors()#FLAGS.word_vector_file)
    print "initializing config"

    vocab_size = extractor.vocab_length + 1
    rap_vec_size = extractor.len_rapper_vector
    num_rappers = extractor.max_nrps
    train_filename = get_train_file()
    valid_filename = get_valid_file()
    test_filename = get_test_file()
    config = get_config(vocab_size, rap_vec_size, num_rappers, max_pron_length)
    eval_gen_config = config.to_eval_gen_config()

    if FLAGS.generate:
        rappers = ["MF Doom"]
        starter = "Fuck everything"
        generate_text(extractor, eval_gen_config, rappers, starter)
        return

    print "initializing session:"
    with tf.Graph().as_default():
        init = initializer()

        with tf.name_scope("Train"):
            print "initializing training model:"
            train_ln_input = LNInput(extractor=extractor, config=config,
                                     filename=train_filename, name="TrainInput",
                                     epoch_size=FLAGS.train_epoch_size)
            with tf.variable_scope("Model", reuse=None, initializer=init):
                m = FullLNModel(is_training=True, config=config, input_=train_ln_input)
            #tf.scalar_summary("Training Loss", m.cost)
            #tf.scalar_summary("Learning Rate", m.lr)
        with tf.name_scope("Valid"):
            print "initializing valid model:"
            valid_ln_input = LNInput(extractor=extractor, config=config,
                                     filename=valid_filename, name="ValidInput",
                                     epoch_size=FLAGS.valid_epoch_size)
            with tf.variable_scope("Model", reuse=True, initializer=init):
                mvalid = FullLNModel(is_training=True, config=config,
                                     input_=valid_ln_input)
            #tf.scalar_summary("Validation Loss", mvalid.cost)
        with tf.name_scope("Test"):
            print "initializing test model:"
            test_ln_input = LNInput(extractor=extractor, config=eval_gen_config,
                                    filename=test_filename, name="TestInput",
                                    epoch_size=FLAGS.test_epoch_size)
            print "TEST EPOCH SIZE:", test_ln_input.epoch_size
            with tf.variable_scope("Model", reuse=True, initializer=init):
                mtest = FullLNModel(is_training=False, config=eval_gen_config,
                                    input_=test_ln_input)


        init_all_op = tf.initialize_all_variables()
        # saver = 0
        # ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
        # if ckpt and ckpt.model_checkpoint_path:
            # saver = optimistic_saver(ckpt.model_checkpoint_path)
        sv_base = tf.train.Supervisor(logdir=FLAGS.save_path, init_op=init_all_op)

        manager = PartialSessionManager(
              local_init_op=sv_base._local_init_op,
              ready_op=sv_base._ready_op,
              ready_for_local_init_op=sv_base._ready_for_local_init_op,
              graph=sv_base._graph,
              recovery_wait_secs=sv_base._recovery_wait_secs)

        sv = PartialSupervisor(logdir=FLAGS.save_path, init_op=init_all_op,
                               session_manager=manager)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=20,
				    log_device_placement=False)
        with sv.managed_session(config=tf_config) as session:
            # if FLAGS.restore_from_checkpoint:
                # # Restore variables from disk.
                # ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
                # if ckpt and ckpt.model_checkpoint_path:
                    # optimistic_restore(session, ckpt.model_checkpoint_path)
                    # #sv.saver.restore(session, ckpt.model_checkpoint_path)
                    # print "Model restored from file " + ckpt.model_checkpoint_path
                # else:
                    # raise Exception("Model ckpt not found")
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, word_vectors, pronunciation_vectors, eval_op=m.train_op,
                                             verbose=True, extractor=extractor)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, word_vectors, pronunciation_vectors)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                sys.stdout.flush()

            test_perplexity = run_epoch(session, mtest, word_vectors, pronunciation_vectors)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

# def optimistic_saver(save_file):
    # reader = tf.train.NewCheckpointReader(save_file)
    # saved_shapes = reader.get_variable_to_shape_map()
    # var_names = sorted([(var.name, var.name.split(':')[0], var.dtype) for var in tf.global_variables()
            # if var.name.split(':')[0] in saved_shapes])
    # restore_vars = []
    # with tf.variable_scope('', reuse=True):
        # for var_name, saved_var_name, dtype in var_names:
            # try:
                # curr_var = tf.get_variable(saved_var_name, dtype=dtype)
            # except ValueError:
                # print "Could not load {}".format(var_name)
            # else:
                # var_shape = curr_var.get_shape().as_list()
                # if var_shape == saved_shapes[saved_var_name]:
                    # restore_vars.append(curr_var)
    # saver = tf.train.Saver(restore_vars)
    # return saver

if __name__ == "__main__":
    tf.app.run()
