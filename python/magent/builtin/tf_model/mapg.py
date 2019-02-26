""" multi-agent policy gradient with centralized critic """

import numpy as np
import tensorflow as tf

from .base import TFBaseModel

class MAPolicyGradient(TFBaseModel):
    def __init__(self, env, handle, name, learning_rate=1e-3,
                 batch_size=64, reward_decay=0.99, eval_obs=None,
                 train_freq=1, value_coef=0.1, ent_coef=0.08):
        """init a model

        Parameters
        ----------
        env: Environment
            environment
        handle: Handle (ctypes.c_int32)
            handle of this group, can be got by env.get_handles
        name: str
            name of this model
        learning_rate: float
        batch_size: int
        reward_decay: float
            reward_decay in TD
        eval_obs: numpy array
            evaluation set of observation
        train_freq: int
            mean training times of a sample
        ent_coef: float
            weight of entropy loss in total loss
        value_coef: float
            weight of value loss in total loss
        """
        TFBaseModel.__init__(self, env, handle, name, "tfmapg")
        # ======================= set config ===================
        self.env = env
        self.handle = handle
        self.name = name
        self.state_space = env.get_state_space()
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.reward_decay = reward_decay
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_freq = train_freq

        self.value_coef = value_coef
        self.ent_coef = ent_coef

        self.train_ct = 0

        # ======================== build network ==================
        with tf.name_scope(self.name):
            self._create_network(self.state_space, self.view_space, self.feature_space)

        # init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # init training buffers
        # self.view_buf = np.empty((1,) + self.view_space)
        # self.feature_buf = np.empty((1,) + self.feature_space)
        # self.action_buf = np.empty(1, dtype=np.int32)
        # self.reward_buf = np.empty(1, dtype=np.float32)

    def _create_network(self, state_space, view_space, feature_space):
        """define computation graph of network

        Parameters
        ----------
        state_space: tuple
        view_space: tuple
        feature_space: tuple
            the input shape
        """
        # input
        n_timestep = None
        n_agents = None
        input_state = tf.placeholder(tf.float32, (n_timestep,) + state_space, name='input_state')
        input_view = tf.placeholder(tf.float32, (n_timestep, n_agents) + view_space, name='input_view') # timestep + n_agents
        input_feature = tf.placeholder(tf.float32, (n_timestep, n_agents) + feature_space, name='input_feature')
        action = tf.placeholder(tf.int32, [n_timestep, n_agents], name='action')
        target = tf.placeholder(tf.float32, [n_timestep], name='target')

        input_shape = tf.shape(input_view)[:2]

        kernel_num = [32, 32]
        hidden_size = [256]

        flatten_view = tf.reshape(input_view, [-1, input_shape[1], np.prod(view_space)], name='flatten_view')
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu, name='h_view')
        h_emb = tf.layers.dense(input_feature, units=hidden_size[0], activation=tf.nn.relu, name='h_emb')

        dense = tf.concat([h_view, h_emb], axis=2)
        dense = tf.layers.dense(dense, units=hidden_size[0] * 2, activation=tf.nn.relu)

        policy = tf.layers.dense(dense, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10, name='policy')

        action_mask = tf.one_hot(action, self.num_actions, name='action_mask')
        action_emb = tf.layers.dense(action_mask, units=hidden_size[0], activation=tf.nn.relu, name='action_emb')
        
        # state emb
        h_conv1 = tf.layers.conv2d(input_state, filters=kernel_num[0], kernel_size=3,
                                   activation=tf.nn.relu, name='h_conv1')
        h_conv2 = tf.layers.conv2d(h_conv1, filters=kernel_num[1], kernel_size=3,
                                   activation=tf.nn.relu, name='h_conv2')
        flatten_state = tf.reshape(h_conv2, [-1, np.prod([v.value for v in h_conv2.shape[1:]])], name='flatten_state')
        h_state = tf.layers.dense(flatten_state, units=hidden_size[0], activation=tf.nn.relu, name='h_state')

        # joint action emb
        dense = tf.concat([h_emb, action_emb], axis=2)
        dense = tf.layers.dense(dense, units=hidden_size[0], activation=tf.nn.relu)
        invariant = tf.reduce_sum(dense, axis=1, name='invariant')
        # value = tf.layers.dense(tf.concat([h_state, invariant], axis=1), units=1)
        # value = tf.reshape(value, (-1,)) # Q(s,a)
        # advantage = tf.stop_gradient(value)
        # TODO make Q(s,a) for credit assignment
        # TODO make local Q-function
        value = tf.layers.dense(h_state, units=1)
        value = tf.reshape(value, (-1,), name='value') # V(s)
        advantage = tf.stop_gradient(target - value, name='advantage')

        log_policy = tf.log(policy + 1e-6, name='log_policy')
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=2, name='log_prob')
        # no advantage for now
        # pg_loss = -tf.reduce_mean(tf.broadcast_to(advantage, input_shape) * log_prob)
        pg_loss = -tf.reduce_mean(tf.broadcast_to(tf.expand_dims(advantage, -1), input_shape) * log_prob, name='pg_loss')

        vf_loss = self.value_coef * tf.reduce_mean(tf.square(target - value), name='vf_loss')
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=2), name='neg_entropy')
        total_loss = tf.identity(pg_loss + vf_loss + neg_entropy, name='total_loss')

        # train op (clip gradient)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.input_state = input_state
        self.input_view = input_view
        self.input_feature = input_feature
        self.action = action
        self.target = target

        self.policy, self.value = policy, value
        self.train_op = train_op
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss

    def infer_action(self, raw_obs, ids, *args, **kwargs):
        """infer action for a batch of agents

        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)
        ids: numpy array
            ids of agents

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        view, feature = raw_obs[0], raw_obs[1]
        n = len(view)
        
        policy = self.sess.run(self.policy, {self.input_view: [view],
                                             self.input_feature: [feature]})
        actions = np.arange(self.num_actions)

        ret = np.empty(n, dtype=np.int32)
        for i in range(n):
            ret[i] = np.random.choice(actions, p=policy[0][i])

        return ret

    def train(self, sample_buffer, print_every=1000):
        """feed new data sample and train

        Parameters
        ----------
        sample_buffer: magent.utility.EpisodesBuffer
            buffer contains samples

        Returns
        -------
        loss: list
            policy gradient loss, critic loss, entropy loss
        value: float
            estimated state value
        """
        assert sample_buffer.use_global_state == True
        sample_buffer.states

        self.view_buf = np.stack([episode.views for episode in sample_buffer.episodes()], axis=1)
        self.feature_buf = np.stack([episode.features for episode in sample_buffer.episodes()], axis=1)
        self.action_buf = np.stack([episode.actions for episode in sample_buffer.episodes()], axis=1)
        self.reward_buf = np.stack([episode.rewards for episode in sample_buffer.episodes()], axis=1)
        self.reward_buf = np.sum(self.reward_buf, axis=1)
        self.state_buf = sample_buffer.states

        # TD target calculation
        # TODO TD-lambda
        values = self.sess.run(self.value, feed_dict={self.input_state: self.state_buf})
        targets = self.reward_buf[:-1] + self.reward_decay * values[1:]

        # train
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value],
            feed_dict={
                self.input_state: self.state_buf[:-1],
                self.input_view: self.view_buf[:-1],
                self.input_feature: self.feature_buf[:-1],
                self.action: self.action_buf[:-1],
                self.target: targets
            }
        )
        print("sample", pg_loss, vf_loss, ent_loss)

        return [pg_loss, vf_loss, ent_loss], np.mean(state_value)

    def get_info(self):
        return ""