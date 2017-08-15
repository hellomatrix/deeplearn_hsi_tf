import tensorflow as tf

class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

if __name__ == '__main__':

    import numpy as np

    import sklearn.preprocessing as prep
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    from Data import Data
    from Data import Data_Set
    import Config
    from fully_connected_hsi_classfier_spatial_feature import fill_feed_dict

    # ## data 1------------------------
    dataname = Config.ksc
    class_num = Config.ksc_class_num
    # ## data 1------------------------

    # set log dir
    ckpt_dir = '/ckpt/'

    pd = Data(dataname)
    data_sets = pd.get_train_valid_test_of_spectral_feature()

    # init data
    train_data = Data_Set([data_sets[0], data_sets[1]])
    valid_data = Data_Set([data_sets[2], data_sets[3]])
    test_data = Data_Set([data_sets[4], data_sets[5]])


    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]


    # X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    # n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 1280
    display_step = 1

    autoencoder = Autoencoder(n_input=176,
                              n_hidden=10,
                              transfer_function=tf.nn.softplus,
                              optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(10000):
            feed_dict = fill_feed_dict(train_data, 'input', 'label')
            batch_xs=feed_dict['input']
            # batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
            cost = autoencoder.partial_fit(batch_xs)
            # Compute average loss
            # avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        # if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
