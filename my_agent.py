from game2048.agents import *
from game2048.displays import *
from game2048.game import *
import tensorflow as tf
import numpy as np

class MyAgent(Agent):
    def __init__(self, game=None, display=None, name="", batch_size=16, sess=None):
        self.name = name
        self.batch_size = batch_size
        self.game = game
        self.display = display
        self.sess = sess
        if game and game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

    def step(self):
        board = self.game.board.reshape(-1, 4, 4)
        board = self._board2onehot(board)

        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('model')
        saver.restore(self.sess, model_file)

        _pred = self.sess.run(self.pred, feed_dict={self.input_data: board})
        _pred = _pred.reshape(4)
        direction = np.argmax(_pred)
        return direction

    def build(self, is_training=False):
        filters = 128
        weight_decay = 0.005
        hidden_units1 = 2048
        hidden_units2 = 512
        drop_prob = 0.3
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 11])

        with tf.variable_scope('conv' + self.name):
            conv41 = tf.layers.conv2d(self.input_data, filters, (4, 1), padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            conv14 = tf.layers.conv2d(self.input_data, filters, (1, 4), padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            conv22 = tf.layers.conv2d(self.input_data, filters, 2, padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            conv33 = tf.layers.conv2d(self.input_data, filters, 3, padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            conv44 = tf.layers.conv2d(self.input_data, filters, 4, padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            concat = tf.concat([conv41, conv14, conv22, conv33, conv44], axis=3)
            conv_2 = tf.layers.conv2d(concat, filters * 5, 3, padding='same', strides=(1, 1),
                                        kernel_regularizer=regularizer, activation=tf.nn.relu)
            feature = tf.reshape(conv_2, [-1, 4 * 4 * filters * 5])
        
        with tf.variable_scope('fc' + self.name):
            fc1 = tf.layers.dense(feature, hidden_units1, tf.nn.relu, kernel_regularizer=regularizer)
            dropout1 = tf.cond(tf.equal(is_training, True), lambda: tf.nn.dropout(fc1, drop_prob), lambda: fc1)
            fc2 = tf.layers.dense(dropout1, hidden_units2, tf.nn.relu, kernel_regularizer=regularizer)
            dropout2 = tf.cond(tf.equal(is_training, True), lambda: tf.nn.dropout(fc2, drop_prob), lambda: fc2)
            self.pred = tf.layers.dense(dropout2, 4, tf.nn.softmax, kernel_regularizer=regularizer)

    def _board2onehot(self, board):
        board = np.log2(np.maximum(board, 1)).astype(np.int32)
        board_onthot = np.zeros((board.shape[0], 4, 4, 11), dtype=np.int32)
        for k in range(board.shape[0]):
            for i in range(4):
                for j in range(4):
                    value = board[k, i, j] if board[k, i, j] < 11 else 10
                    board_onthot[k, i, j, value] = 1
        return board_onthot

    def train(self, learning_rate=1e-6):
        train_games = 5000
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        with tf.name_scope('cal_loss' + self.name):
            cross_entropy = -tf.reduce_sum(self.input_label * tf.log(self.pred))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = cross_entropy + regularization_loss

        with tf.name_scope('acc' + self.name):
            predition = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.input_label, 1))
            acc = tf.reduce_mean(tf.cast(predition, tf.float32))
        
        with tf.name_scope('optimizer' + self.name):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, model_file)

        self.sess = sess

        for cnt in range(train_games):
            games = [Game(4, score_to_win=1024, random=False) for _ in range(self.batch_size)]
            agents = [ExpectiMaxAgent(game) for game in games]
            direction = [0] * self.batch_size
            
            flag = True
            while flag:
                data_batch = np.empty((self.batch_size, 4, 4), dtype=np.int32)
                label_batch = np.zeros((self.batch_size, 4), dtype=np.int32)
                for i in range(self.batch_size):
                    direction[i] = agents[i].step()
                    label_batch[i, direction[i]] = 1
                    data_batch[i] = games[i].board
                data_batch = self._board2onehot(data_batch)
                _loss, _acc, _ = sess.run([loss, acc, optimizer], feed_dict={
                    self.input_data: data_batch,
                    self.input_label: label_batch
                })
                for i in range(self.batch_size):
                    games[i].move(direction[i])
                    if games[i].end:
                        flag = False
                        break
            
            if (cnt + 1) % 50 == 0:
                saver.save(sess, 'model/model.ckpt', global_step=cnt+1)


        saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    agent = MyAgent(display=Display())
    agent.build(is_training=True)
    agent.train()
