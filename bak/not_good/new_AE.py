
import tensorflow as tf
import Config
import pdb

class new_AE(object):

    def __init__(self,img_placeholder,layer_units,lr=Config.learn_rate,sess=None):

        self.feature_dim = int(img_placeholder.shape[1])
        self.layer_units = layer_units
        self.transfer = tf.nn.sigmoid ## ?????  what is this
        optimizer = tf.train.GradientDescentOptimizer(lr)
        self.logits = None


        net_weights = self.initialize_weights()
        self.weights = net_weights

        tf.summary.histogram('AE_weights',self.weights['w1'])
        tf.summary.histogram('AE_weights',self.weights['b1'])

        #model self.x is the label of input
        self.x = img_placeholder
        #self.x = tf.placeholder(tf.float32,[None,self.feature_dim]) ## important to save real data

        self.hidden = self.transfer(tf.add(tf.matmul(self.x,self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #cost
        self.cost = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(self.reconstruction,self.x))))

        self.optimizer = optimizer.minimize(self.cost)


        self.sess = sess
        # init = tf.global_variables_initializer()
        # # self.sess = tf.Session()
        # self.sess.run(init)

    def initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable('w1',shape=[self.feature_dim,self.layer_units],
                                             initializer=tf.random_normal_initializer(mean=0, stddev=1.0))

        all_weights['b1'] = tf.Variable(tf.zeros([self.layer_units],dtype=tf.float32),name='b1')
        all_weights['w2'] = tf.identity(tf.transpose(all_weights['w1']),name='w2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_dim], dtype=tf.float32),name='b2')

        return all_weights

    def partial_fit(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X}) # the same feature dim

    def cal_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X})

    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X})

    def getWeigths(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

if __name__ == '__main__':

    import numpy as np
    import time
    import Config
    from Data import Data
    from Data import Data_Set
    from fully_connected_hsi_classfier_spatial_feature import fill_feed_dict


    # ## data 1------------------------
    dataname = Config.ksc
    class_num = Config.ksc_class_num
    # ## data 1------------------------

    # set log dir
    ckpt_dir = '/ckpt/'

    with tf.Graph().as_default() as gd:

        pd = Data(dataname)
        data_sets = pd.get_train_valid_test_of_spectral_feature()

        # init data
        train_data = Data_Set([data_sets[0],data_sets[1]])
        valid_data = Data_Set([data_sets[2],data_sets[3]])
        test_data =  Data_Set([data_sets[4],data_sets[5]])

        with tf.Session(graph=gd) as sess:

            img_placeholder = tf.placeholder(tf.float32, [None, data_sets[0].shape[1]])  ## important to save real data
            ae = new_AE(img_placeholder, layer_units=60, sess=sess)
            saver = tf.train.Saver()  # after graph
            init = tf.global_variables_initializer() # order is important,after graph
            sess.run(init)  #after graph

            tf.summary.scalar('loss', ae.cost) # must be a tensor or opration under tensorflow, ae.cost is ok, but ae.cal_total_cost is wrong
            tf.summary.histogram('w1',ae.getWeigths())
            tf.summary.histogram('b1',ae.getBiases())

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./AE_graph/',sess.graph)

            for step in range(10000):
                start_time = time.time()
                feed_dict = fill_feed_dict(train_data, 'input', 'label')

                cost = ae.partial_fit(feed_dict['input'])

                duration = time.time() - start_time
                # pdb.set_trace()
                if step % 1000 == 0:
                    print('Step %d: loss = %.2f(%.3f sec)' % (step, cost, duration))
                    summary_ = sess.run(merged, {ae.x: feed_dict['input']}) # call ae.cost, so we need give the operation a diction
                    writer.add_summary(summary_,step)
                    saver.save(sess,'./ckpt/model.ckpt',global_step=step)


