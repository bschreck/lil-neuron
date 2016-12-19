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


def initializer():
    return tf.contrib.layers.variance_scaling_initializer(dtype=data_type())


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
        self.verse_length = input_data["words"].get_shape()[:1]
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
    def __init__(self, words, verse_lengths, context, context_size):
        self.words = words
        self.verse_lengths = verse_lengths
        self.context = context
        self.context_size = context_size


class RNNPath(object):
    def __init__(self, is_training, config, input_, device=FLAGS.device):
        self.device = device
        self._input = input_
        words = input_.words
        verse_lengths = input_.verse_lengths
        self.context = input_.context
        self.context_size = input_.context_size

        self.batch_size = config.batch_size
        self.rnn_size = config.hidden_size
        self.num_layers = config.num_layers
        self.embedding_dim = config.embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.vocab_size = config.vocab_size

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
            cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, **lstm_args)

            if is_training and config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=config.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers,
                                               state_is_tuple=True)

            self._initial_state = cell.zero_state(self.batch_size, data_type())



            embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
                trainable=False, name="word_vector")

            embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])
            embedding_init = embedding.assign(embedding_placeholder)

            pron_lookup = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, config.max_pron_length]),
                trainable=False, name="pron_lookup")

            pron_lookup_placeholder = tf.placeholder(tf.float32, [self.vocab_size, config.max_pron_length])
            pron_lookup_init = pron_lookup.assign(pron_lookup_placeholder)



            pronunciation_embedding = tf.get_variable("pronunciation_embedding_W",
                    initializer=initializer(),
                    shape=[config.max_pron_length, self.embedding_dim],
                    trainable=True,
                    dtype=data_type())
            pronunciation_b = tf.get_variable("pronunciation_embedding_b",
                                              initializer=initializer(),
                                              shape=[self.embedding_dim],
                                              trainable=True,
                                              dtype=data_type())

            context_embedding = tf.get_variable("context_embedding_W",
                    initializer=initializer(),
                    shape=[self.context_size, self.context_embedding_dim],
                    trainable=True,
                    dtype=data_type())
            context_b = tf.get_variable("context_embedding_b",
                                              initializer=initializer(),
                                              shape=[self.context_embedding_dim],
                                              trainable=True,
                                              dtype=data_type())


            word_embed_inputs = tf.nn.embedding_lookup(embedding, words)
            pron_inputs = tf.nn.embedding_lookup(pron_lookup, words)
            pron_embed_inputs = tf.nn.batch_matmul(pronunciation_embedding, pron_inputs) + pronunciation_b
            context_embed_inputs = tf.nn.batch_matmul(context_embedding, self.context) + context_b
            # TODO: try dropout, nonlinearity



            combined = tf.concat(2, [word_embed_inputs, pron_embed_inputs, context_embed_inputs])

            resize_layer = tf.get_variable("rnn_resize",
                                           initializer=initializer(),
                                           shape=[2*self.embedding_dim + self.context_embedding_dim, config.rnn_size],
                                           trainable=True,
                                           dtype=data_type())
            rnn_input = tf.nn.batch_matmul(combined, resize_layer)


            if is_training and config.keep_prob < 1:
                rnn_input = tf.nn.dropout(rnn_input, config.keep_prob)

            verse_lengths = tf.reshape(verse_lengths, [-1])
            verse_lengths = tf.tile(verse_lengths, [self.batch_size])
            outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                     dtype=data_type(),
                                                     sequence_length=verse_lengths,
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


class Learn(object):
    def __init__(self, is_training, config, labels, input, verse_lengths,
                 device=FLAGS.device):
        self.rnn_path = input
        self.context = input.context
        self.context_size = input.context_size
        self.vocab_size = config.vocab_size

        with tf.device(device):
            verse_len = tf.shape(self.rnn_path.outputs)[1]

            concat = tf.concat(2, [self.rnn_path.outputs, self.context])
            outputs_flat = tf.reshape(self.rnn_path.outputs, [-1, self.rnn_path.output_size + self.context_size])

            final_unit_size = self.rnn_path.output_size + self.context_size

            softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=data_type())
            print "vocab size:", self.vocab_size
            softmax_W = tf.get_variable(
                name="softmax_w",
                initializer=initializer(),
                shape=[final_unit_size, self.vocab_size],
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
        context = tf.concat(1, rappers)

        labels = input_.labels

        rnn_input = RNNInput(words=input_.words,
                             context=context,
                             context_size = num_rappers * self.config.rap_vec_size,
                             verse_lengths=self.verse_length)
        with tf.variable_scope("RNNPath"):
            rnn_path = RNNPath(is_training=is_training,
                               input_=rnn_input,
                               config=config,
                               device=device)


        with tf.variable_scope("Learn"):
            self.learn = Learn(is_training=is_training,
                               labels=labels,
                               input=rnn_path,
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
        return self.learn.initial_state

    @property
    def final_state(self):
        return self.learn.final_state

    @property
    def lr(self):
        return self.learn.lr

    @property
    def train_op(self):
        return self.learn.train_op

class Config(object):
    def __init__(self, vocab_size, rap_vec_size, num_rappers):
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
    num_layers = 2
    max_num_steps = 20
    hidden_size = 200
    embedding_dim = 300
    context_embedding_dim = 300
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
    max_num_steps = 35
    hidden_size = 650
    embedding_dim = 300
    context_embedding_dim = 300
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20


class LargeConfig(Config):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    max_num_steps = 35
    hidden_size = 1500
    embedding_dim = 300
    context_embedding_dim = 300
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20


class TestConfig(Config):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 2
    max_num_steps = 2
    hidden_size = 2
    embedding_dim = 300
    context_embedding_dim = 300
    max_epoch = 1
    max_max_epoch = 1
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

    session.run([model.embedding_init, model.pron_lookup_init],
                feed_dict={model.embedding_placeholder: model.embedding_array,
                           model.pron_lookup_placeholder: model.pron_lookup_array})

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

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
            words = ["<eos>", "<eov>", "<eor>"]
            words.append("<nrp:{}>".format(';'.join(rappers)))
            starter = starter.replace(". ", " <eos> ").replace('\n', ' ').split()
            words.extend(starter)
            tensor_dict, input_data, init_op_local = extractor.gen_features_from_starter(rappers, words)

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
    rap_vec_size = extractor.len_rapper_vector
    num_rappers = extractor.max_nrps
    train_filename = get_train_file()
    valid_filename = get_valid_file()
    test_filename = get_test_file()
    config = get_config(vocab_size, rap_vec_size, num_rappers)
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
