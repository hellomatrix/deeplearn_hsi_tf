
import tensorflow as tf
from new_AE import new_AE
import time
import Config

from Data import fill_feed_dict
import os


def reuse_layer(temp_img_placeholder, layer_units,trans_function=tf.nn.softplus):

        feature_dim = int(temp_img_placeholder.shape[1])
        weights = tf.get_variable('w1', shape=[feature_dim, layer_units],
                                  initializer=tf.random_normal_initializer(mean=0, stddev=1.0))
        biases = tf.Variable(tf.zeros([layer_units],dtype=tf.float32),name='b1')
        rc = tf.add(tf.matmul(temp_img_placeholder, weights) , biases)
        h = trans_function(rc)
        print('operation name of reuse layer:%s'%h)

        return h,rc


class new_SAE(object):

    def __init__(self,input_dim,encoder_shape,transfer_function=tf.nn.softplus,optimizer = tf.train.AdamOptimizer()):

        self.encoder_W = []
        self.encoder_b = []
        self.encoder_shape = encoder_shape
        self.feature_dim = input_dim[0]
        self.labels_dim = input_dim[1]
        self.encoder_layers=[]
        self.encoder_cost = []
        self.decoder_weights = []
        self.final_summayrs =[]
        self.sae_summayrs = []

        self.transfer_function=transfer_function
        self.optimizer = optimizer

        # input
        self.x = tf.placeholder(tf.float32,[self.labels_dim,self.feature_dim]) ## important to save real data
        self.y = tf.placeholder(tf.float32,[self.labels_dim]) ## important to save real data


        temp_input_dim = self.feature_dim
        temp_img_placeholder = self.x

        encoders_reconstruction = None

        self.encoder_h=[]
        self.decoder_h=[] #out_puts of decoder

        #encoder layers
        for i in range(len(self.encoder_shape )):
            print('encoder: i=%d'%i)

            with tf.variable_scope('encode_layer{0}'.format(i)):
                ae = new_AE(temp_input_dim, encoder_shape[i])

                temp_input_dim = encoder_shape[i]
                print('operation name of AE layer:%s' %ae.hidden)

                # temp_cost = tf.reduce_mean(tf.pow(tf.subtract(temp_ae.reconstruction, self.x), 2.0))
                self.encoder_cost.append(ae.cost)
                self.encoder_layers.append(ae)

            # the real encoder layers in encoder
            with tf.variable_scope('encode_layer{0}'.format(i),reuse=True):

                temp_img_placeholder, encoders_reconstruction = \
                    reuse_layer(temp_img_placeholder,layer_units=encoder_shape[i],trans_function=self.transfer_function)

                self.encoder_h.append(temp_img_placeholder)

        self.encoders_reconstruction = encoders_reconstruction

        # decoder layers
        h = temp_img_placeholder
        for i in range(len(self.encoder_shape)):
            print('decoder :i=%d'%i)
            with tf.variable_scope('decode_layer{0}'.format(i)):
                ae = self.encoder_layers[-(i+1)]

                w2 = tf.identity(ae.weights['w2'], name='w2_{0}'.format(4-i))
                b2 = tf.identity(ae.weights['b2'], name='b2_{0}'.format(4-i))

                # print(ae,w2,b2,h)

                if i == 3:
                    h = tf.add(tf.matmul(h, w2), b2)

                else:
                    h = self.transfer_function(tf.add(tf.matmul(h, w2), b2))

                self.decoder_h.append(h)
                self.decoder_weights.append(w2)

        self.reconstruction = h

        #cost
        #self.cost = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(self.reconstruction,self.x))))
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        #train
        self.optimizer = self.optimizer.minimize(self.cost)


        # final model cost
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.y,tf.int32), logits=self.encoders_reconstruction, name='xentropy')

        self.cost_final = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        # Create a optimization method
        optimizer_final = tf.train.GradientDescentOptimizer(Config.final_learn_rate)

        # Set the trainning objection to minimize loss
        self.optimizer_final = optimizer_final.minimize(self.cost_final)


        # correct numbers
        correct = tf.nn.in_top_k(self.encoders_reconstruction, tf.cast(self.y,tf.int32),
                                 1)  # if the maxmum k value could match labels, return True
        self.correct = tf.reduce_sum(tf.cast(correct, tf.int32))
fdafdsafsdfsdf

        # #----------tensorboard-------------------------------------------------
        with tf.name_scope('sae_summary') as sae_smry:

            self.sae_loss_scaler = tf.summary.scalar('loss',self.cost)

            # self.sae_hist = tf.summary.histogram('encoder_weights', self.weights['w1'])
            #
            # im_w = self.weights['w1'] * 100
            # im_w = tf.expand_dims(tf.expand_dims(im_w, 0), -1)
            # self.sae_hist_img_weight = tf.summary.image('AE_weights', im_w)

            self.sae_summayrs.append(self.sae_loss_scaler)

            # self.sae_summayrs.append(self.sae_hist)
            # self.sae_summayrs.append(self.sae_img_weight)

            self.sae_merged = tf.summary.merge(self.sae_summayrs,sae_smry)

            # tf.summary.histogram('AE_weights',self.weights['b1'])

            # im_w_max = tf.cast(tf.arg_max(tf.arg_max(self.weights['w1'],0),1),tf.float32)
            # im_w_min = tf.cast(tf.arg_min(tf.arg_min(self.weights['w1'],0),1),tf.float32)
            # im_w = tf
            # .multiply(tf.divide(tf.subtract(self.weights['w1'],im_w_min),tf.subtract(im_w_max,im_w_min)),255)

        # #----------tensorboard-------------------------------------------------

        # #----------tensorboard-------------------------------------------------
        with tf.name_scope('final_model_summary') as f_smry:

            self.final_loss_scaler = tf.summary.scalar('loss', self.cost_final)

            # self.sae_hist = tf.summary.histogram('encoder_weights', self.weights['w1'])
            #
            # im_w = self.weights['w1'] * 100
            # im_w = tf.expand_dims(tf.expand_dims(im_w, 0), -1)
            # self.sae_hist_img_weight = tf.summary.image('AE_weights', im_w)

            self.final_summayrs.append(self.final_loss_scaler)

            # self.sae_summayrs.append(self.sae_hist)
            # self.sae_summayrs.append(self.sae_img_weight)

            self.final_merged = tf.summary.merge(self.final_summayrs, f_smry)



    def transform(self,feed_dict,layer_idx,sess):
        return sess.run(self.encoder_h[layer_idx-1], feed_dict=feed_dict) # do not use new new_AE object


    # pre train the AE and SAE
    def pre_train(self,train_data,sess,path):

        # train AE
        epoch_size = train_data._num_examples
        writer = tf.summary.FileWriter(path,tf.get_default_graph())

        for i in range(len(self.encoder_layers)):
            print('pre train: i=%d' % i)
            ae = self.encoder_layers[i]

            for step in range(int(Config.epoch_ae_pretrain_times*epoch_size/Config.batch_size)):
                start_time = time.time()
                feed_dict = fill_feed_dict(train_data, 'input', 'label')
                X = feed_dict['input']

                if i > 0:
                    X = self.transform(feed_dict={self.x: X}, layer_idx=i, sess=sess)
                else:
                    pass

                cost, _ = ae.partial_fit(feed_dict={ae.x: X}, sess=sess)
                duration = time.time() - start_time

                if step % 1000 == 0:
                    print('encoder laye%d, Step %d, loss = %.2f(%.3f sec)' % (i, step, cost, duration))
                    summary_ = sess.run(ae.merged, feed_dict={ae.x: X})
                    writer.add_summary(summary_,step)
                    # saver.save(sess,os.path.join(ckpt_dir,'mode.ckpt'),global_step=step)

        # train SAE
        for step in range(int(Config.epoch_sae_pretrain_times*epoch_size/Config.batch_size)):
            start_time = time.time()
            feed_dict = fill_feed_dict(train_data, 'input', 'label')
            X = feed_dict['input']

            cost, _ = self.train_SAE(feed_dict={self.x: X}, sess=sess)
            duration = time.time() - start_time

            if step % 1000 == 0:
                print('train SAE: Step %d: loss = %.2f(%.3f sec)' % (step, cost, duration))
                summary_ = sess.run(self.sae_merged, feed_dict={self.x: X})
                writer.add_summary(summary_, step)
                # saver.save(sess, os.path.join(ckpt_dir, 'pretrain_SAE.ckpt'), global_step=step)


    def train_SAE(self,feed_dict,sess):
        cost, op = sess.run((self.cost,self.optimizer), feed_dict=feed_dict)  # the same feature dim
        return cost, op


    def getWeigths_d(self,sess):
        return sess.run(self.decoder_weights)


    def train_final_model(self,feed_dict,sess):

        return sess.run([self.cost_final,self.optimizer_final],feed_dict=feed_dict)




if __name__ == '__main__':

    import numpy as np
    import time
    import Config
    from Data import Data
    from Data import Data_Set

    # ## data 1------------------------
    dataname = Config.paviaU
    # ## data 1------------------------

    # set log dir
    ckpt_dir = './SAE_ckpt/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with tf.Graph().as_default() as gr:

        pd = Data(dataname)
        data_sets = pd.get_train_valid_test_of_spectral_feature()

        # init data
        train_data = Data_Set([data_sets[0],data_sets[1]])
        valid_data = Data_Set([data_sets[2],data_sets[3]])
        test_data = Data_Set([data_sets[4],data_sets[5]])


        with tf.Session(graph=gr) as sess:
            # img_placeholder = tf.placeholder(tf.float32,[None,data_sets[0].shape[1]]) ## important to save real data
            # img_placeholder = tf.placeholder(tf.float32,[None,3]) ## important to save real data

            input_dim = [train_data.feature_dim,Config.batch_size]

            sae = new_SAE(input_dim = input_dim, encoder_shape=[60,60,60,60])

            saver = tf.train.Saver()  # after graph, sometime with variable name ,so so
            init = tf.global_variables_initializer() # order is important,after graph
            sess.run(init)  #after graph

            sae.pre_train(train_data, sess)

# # #---------------------------------------------------------------
#             # sae_s=tf.summary.scalar('SAE_loss', sae.cost)
#
#             # merged_SAE = tf.summary.merge(sae_s)
#             #merged_SAE = tf.summary.merge_all()5
#             writer = tf.summary.FileWriter('./SAE_graph/',gr)
#
#             # pre_train
#             # merged_AE = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES),pc)
#             for i in range(len(sae.encoder_layers)):
#                 print('pre train: i=%d'%i)
#                 ae = sae.encoder_layers[i]
#                 # tf.summary.scalar('loss',ae.cost)
#                 for step in range(1000):
#                     start_time = time.time()
#                     feed_dict = fill_feed_dict(train_data, 'input', 'label')
#                     X = feed_dict['input']
#                     # X = feed_dict['input'][1:2,0:3]
#                     sae.train_final_model(feed_dict={sae.x:X,sae.y:feed_dict['label']},sess=sess)
#
#                     if i>0:
#                         X = sae.transform(feed_dict={sae.x:X},layer_idx=i,sess=sess)
#                     else:
#                         pass
#
#                     cost,_ = ae.partial_fit(feed_dict={ae.x:X},sess=sess)
#                     duration = time.time() - start_time
#
#                     if step % 100 == 0:
#                         print('loo %d Step %d: loss = %.2f(%.3f sec)' % (i,step, cost, duration))
#                         summary_ = sess.run(ae.merged, feed_dict={ae.x: X})
#                         writer.add_summary(summary_,i)
#                         # saver.save(sess,os.path.join(ckpt_dir,'mode.ckpt'),global_step=step)
#
#             # fine_tune
#             merged_SAE = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
#             tf.summary.scalar('sae_loss', sae.cost)
#
#             for step in range(100000):
#                 start_time = time.time()
#                 feed_dict = fill_feed_dict(train_data, 'input', 'label')
#                 X = feed_dict['input']
#
#                 cost,_= sae.train_SAE(feed_dict={sae.x:X},sess=sess)
#                 duration = time.time() - start_time
#
#                 if step % 100 == 0:
#                     print('fine tune : Step %d: loss = %.2f(%.3f sec)' % (step, cost, duration))
#                     #summary_ = sess.run(merged_SAE, feed_dict={sae.x: X})
#                     #writer.add_summary(summary_,step)
#                     #saver.save(sess, os.path.join(ckpt_dir, 'mode.ckpt'), global_step=step)
#
#     # debug----------------------------------------------------------------------------------
#                         # print(sae.encoder_layers[0].getWeigths())
#                         # print('------------')
#                         # print(np.transpose(sae.getWeigths_d()[1]))
#                         #
#                         # print('diff of encoder weigth and the transpose decoder weights: %f'
#                         #       %np.sum(np.sum((sae.encoder_layers[0].getWeigths()-np.transpose(sae.getWeigths_d()[1])))))
#     # debug----------------------------------------------------------------------------------
#
#                         # summary_ = sess.run(merged, {ae.x: X})
#                         # writer.add_summary(summary_, i)
#                         # saver.save(sess, './SAE_log/mode.ckpt', global_step=step)