import tensorflow as tf
import logging
import time

class PartialSessionManager(tf.train.SessionManager):
    def _restore_checkpoint(self,
                          master,
                          saver=None,
                          checkpoint_dir=None,
                          checkpoint_filename_with_path=None,
                          wait_for_checkpoint=False,
                          max_wait_secs=7200,
                          config=None,
                          sess=None):
        self._target = master
        if sess is None:
            sess = tf.Session(self._target, graph=self._graph, config=config)

        if checkpoint_dir and checkpoint_filename_with_path:
            raise ValueError("Can not provide both checkpoint_dir and "
                             "checkpoint_filename_with_path.")
        # If either saver or checkpoint_* is not specified, cannot restore. Just
        # return.
        if not saver or not (checkpoint_dir or checkpoint_filename_with_path):
            return sess, False

        if checkpoint_filename_with_path:
            saver.restore(sess, checkpoint_filename_with_path)
            return sess, True

        # Waits up until max_wait_secs for checkpoint to become available.
        wait_time = 0
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        while not ckpt or not ckpt.model_checkpoint_path:
            if wait_for_checkpoint and wait_time < max_wait_secs:
                logging.info("Waiting for checkpoint to be available.")
                time.sleep(self._recovery_wait_secs)
                wait_time += self._recovery_wait_secs
                ckpt = tf.get_checkpoint_state(checkpoint_dir)
            else:
                return sess, False

        # Loads the checkpoint.
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        return sess, True

    def prepare_session(self,
                      master,
                      init_op=None,
                      saver=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      wait_for_checkpoint=False,
                      max_wait_secs=7200,
                      config=None,
                      init_feed_dict=None,
                      init_fn=None):
        sess = tf.Session(self._target, graph=self._graph, config=config)
        if init_op is not None:
            sess.run(init_op, feed_dict=init_feed_dict)
        if init_fn:
            init_fn(sess)

        sess, is_loaded_from_checkpoint = self._restore_checkpoint(
            master,
            saver,
            checkpoint_dir=checkpoint_dir,
            checkpoint_filename_with_path=checkpoint_filename_with_path,
            wait_for_checkpoint=wait_for_checkpoint,
            max_wait_secs=max_wait_secs,
            config=config,
            sess=sess)

        local_init_success, msg = self._try_run_local_init_op(sess)
        if not local_init_success:
          raise RuntimeError(
              "Init operations did not make model ready for local_init.  "
              "Init op: %s, init fn: %s, error: %s" % ("None" if init_op is None
                                                       else init_op.name, init_fn,
                                                       msg))

        is_ready, msg = self._model_ready(sess)
        if not is_ready:
          raise RuntimeError(
              "Init operations did not make model ready.  "
              "Init op: %s, init fn: %s, local_init_op: %s, error: %s" %
              (None if init_op is None else init_op.name, init_fn,
               self._local_init_op, msg))
        return sess

class PartialSupervisor(tf.train.Supervisor):
    def __init__(self, *args, **kwargs):
        self.logdir = kwargs.get('logdir')
        super(PartialSupervisor, self).__init__(*args, **kwargs)

    def _init_saver(self, saver=tf.train.Supervisor.USE_DEFAULT):
        """Initializes saver.
        Args:
          saver: A `Saver` object. If set to USE_DEFAULT, create one that
            saves all the variables.
        """
        if saver is tf.train.Supervisor.USE_DEFAULT:
            restore_vars = self.get_restore_vars()

            saver = self._get_first_op_from_collection(tf.GraphKeys.SAVERS)
            try:
                all_vars = tf.global_variables()
            except AttributeError:
                all_vars = tf.all_variables()
            if saver is None and all_vars:
                if restore_vars is not None:
                    saver = tf.train.Saver(restore_vars)
                else:
                    saver = tf.train.Saver()

                tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        self._saver = saver

    def get_restore_vars(self):
        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            save_path = ckpt.model_checkpoint_path
        else:
            return None
        reader = tf.train.NewCheckpointReader(save_path)
        saved_shapes = reader.get_variable_to_shape_map()
        try:
            all_vars = tf.global_variables()
        except AttributeError:
            all_vars = tf.all_variables()
        var_names = sorted([(var.name, var.name.split(':')[0], var.dtype) for var in all_vars
                if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name, dtype in var_names:
                try:
                    curr_var = tf.get_variable(saved_var_name, dtype=dtype)
                except ValueError:
                    print "Could not load {}".format(var_name)
                else:
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)
        return restore_vars
