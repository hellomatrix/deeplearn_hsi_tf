
import tensorflow as tf
import Config
import os

class new_AE(object):

    def __init__(self,input_dim,layer_units,transfer_function=None,optimizer = tf.train.AdamOptimizer()):

        self.feature_dim = input_dim
        self.layer_units = layer_units
        self.summayrs = []
        # transfer_function=tf.nn.softplus
        self.transfer = transfer_function ##  amazing!!!!

        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.AdamOptimizer()

        # self.logits = None
        net_weights = self.initialize_weights()
        self.weights = net_weights

        #model self.x is the label of input
        self.x = tf.placeholder(tf.float32,[None,self.feature_dim]) ## important to save real data

        self.hidden = self.transfer(tf.add(tf.matmul(self.x,self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #cost
        self.cost = 0.5*tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        #cost
        # self.cost = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(self.reconstruction,self.x))))

        self.optimizer = optimizer.minimize(self.cost)

        # #----------tensorboard-------------------------------------------------
        with tf.name_scope('ae_summary') as asummary:

            self.loss_scaler = tf.summary.scalar('loss',self.cost)
            self.hist = tf.summary.histogram('encoder_weights', self.weights['w1'])

            im_w = self.weights['w1'] * 100
            im_w = tf.expand_dims(tf.expand_dims(im_w, 0), -1)
            self.img_weight = tf.summary.image('AE_weights', im_w)

            self.summayrs.append(self.loss_scaler)
            self.summayrs.append(self.hist)
            self.summayrs.append(self.img_weight)

            self.merged = tf.summary.merge(self.summayrs,asummary)

            # tf.summary.histogram('AE_weights',self.weights['b1'])

            # im_w_max = tf.cast(tf.arg_max(tf.arg_max(self.weights['w1'],0),1),tf.float32)
            # im_w_min = tf.cast(tf.arg_min(tf.arg_min(self.weights['w1'],0),1),tf.float32)
            # im_w = tf
            # .multiply(tf.divide(tf.subtract(self.weights['w1'],im_w_min),tf.subtract(im_w_max,im_w_min)),255)

        # #----------tensorboard-------------------------------------------------

    def initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable('w1',shape=[self.feature_dim,self.layer_units],
                                             initializer=tf.random_normal_initializer(mean=0, stddev=1.0))

        all_weights['b1'] = tf.Variable(tf.zeros([self.layer_units],dtype=tf.float32),name='b1')

        all_weights['w2'] = tf.identity(tf.transpose(all_weights['w1']),name='w2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_dim], dtype=tf.float32),name='b2')

        return all_weights

    def partial_fit(self,feed_dict,sess):
        cost,op = sess.run((self.cost,self.optimizer),feed_dict=feed_dict) # the same feature dim
        return cost,op

    def cal_total_cost(self,feed_dict,sess):
        return sess.run(self.cost,feed_dict=feed_dict)

    def reconstruct(self,feed_dict,sess):
        return sess.run(self.reconstruction,feed_dict=feed_dict)

    def transform(self,feed_dict,sess):
        return sess.run(self.hidden,feed_dict=feed_dict)

    def getWeigths(self,sess):
        return sess.run(self.weights['w1'])

    def getBiases(self,sess):
        return sess.run(self.weights['b1'])

    def getWeigths_d(self,sess):
        return sess.run(self.weights['w2'])

    def getBiases_d(self,sess):
        return sess.run(self.weights['b2'])


if __name__ == '__main__':

    import numpy as np
    import time
    import Config
    from Data import Data
    from Data import Data_Set
    from Data import fill_feed_dict


    # ## data 1------------------------
    dataname = Config.paviaU
    # ## data 1------------------------

    # set log dir
    ckpt_dir = './AE_ckpt/'
    needTrain=True
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with tf.Graph().as_default() as gd:

        pd = Data(dataname)
        data_sets = pd.get_train_valid_test_of_spectral_feature()

        # init data
        train_data = Data_Set([data_sets[0],data_sets[1]])
        valid_data = Data_Set([data_sets[2],data_sets[3]])
        test_data =  Data_Set([data_sets[4],data_sets[5]])

        img_placeholder = tf.placeholder(tf.float32, [None, data_sets[0].shape[1]])  ## important to save real data
        # img_placeholder = tf.placeholder(tf.float32,[None,3]) ## important to save real data
        ae = new_AE(train_data._images.shape[1], layer_units=60)

        saver = tf.train.Saver()  # after graph

        # tf.summary.scalar('loss', ae.cost) # must be a tensor or opration under tensorflow, ae.cost is ok, but ae.cal_total_cost is wrong
        # merged = tf.summary.merge_all()

        with tf.Session(graph=gd) as sess:

            init = tf.global_variables_initializer() # order is important,after graph
            sess.run(init)  #after graph

            if needTrain:
                writer = tf.summary.FileWriter('./AE_graph/', sess.graph)

                for step in range(100000):
                    start_time = time.time()
                    feed_dict = fill_feed_dict(train_data, 'input', 'label')
                    X = feed_dict['input']
    # debug----------------------------------------------------------------------------------

                    # X= feed_dict['input'][:,0:3]
                    # print('train data:++++++++++++++++++++++++++++++++++++')
                    # print

                    # print('input data ------------------')
                    ## print(feed_dict['input'][1,1],feed_dict['input'].shape)
                    ## success

    # debug----------------------------------------------------------------------------------
                    cost,op = ae.partial_fit(feed_dict={ae.x:X},sess=sess)
                    duration = time.time() - start_time
                    # pdb.set_trace()
                    if step % 500 == 0:
                       print('Step %d: loss = %.2f(%.3f sec)' % (step, cost, duration))

                       summary_ = sess.run(ae.merged,
                                           feed_dict={ae.x: X})  # call ae.cost, so we need give the operation a diction
                       writer.add_summary(summary_,step)
                       checkpoint_file = os.path.join(ckpt_dir,'train_model.ckpt')
                       saver.save(sess,checkpoint_file,global_step=step)

     # debug----------------------------------------------------------------------------------
                        # print(step,duration,cost)
                        # print('weight:====================================')
                        # print(ae.getWeigths()[0:1,0:5])
                        #
                        # print('train data:++++++++++++++++++++++++++++++++++++')
                        # print(X[1, 1:3])
                        # print('reconstruct:====================================')
                        # print(ae.reconstruct(X)[1,1:3])
                        # # print(ae.getWeigths())
                        # print('COST:====================================')
                        # print(ae.cal_total_cost(X))

                       # print('diff of encoder weigth and the transpose decoder weights: %f'
                       #       %np.sum(np.sum((ae.getWeigths()-np.transpose(ae.getWeigths_d())))))
                       # assert ae.getWeigths() == np.transpose(ae.getWeigths_d())
    # debug----------------------------------------------------------------------------------
            else:
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                saver.restore(sess,ckpt.model_checkpoint_path)

                v = tf.get_collection(tf.GraphKeys.VARIABLES)
                # s = tf.get_collection(tf.GraphKeys.SUMMARIES)


# debug----------------------------------------------------------------------------------

                # print(v,v[0])
                # print(sess.run(v[2]),sess.run(v[0]))
                # print('diff of encoder weigth and the transpose decoder weights: %f'
                #       % np.sum(np.sum((ae.getWeigths(sess) - np.transpose(ae.getWeigths_d(sess))))))


# debug----------------------------------------------------------------------------------
