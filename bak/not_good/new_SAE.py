
import tensorflow as tf
from new_AE import new_AE


# def one_layer(input, layer_units, init=None):
#     feature_dim = int(img_batch.shape[1])
#
#         weights = tf.get_variable('w1', shape=[feature_dim, layer_units],
#                                   initializer=tf.random_normal_initializer(mean=0, stddev=1.0))
#
#         biases = tf.get_variable('b1', shape=[layer_units], initializer=tf.constant_initializer(0))
#
#         h = tf.nn.sigmoid(tf.matmul(img_batch, weights) + biases)
#
#         return h, weights, biases


class new_SAE(object):

    def __init__(self,img_placeholder,encoder_shape,sess=None):

        self.encoder_W = []
        self.encoder_b = []
        self.encoder_shape = encoder_shape
        self.feature_dim = int(img_placeholder.shape[1])
        self.encoder_layers=[]
        self.encoder_cost = []
        self.sess = sess
        # input
        self.x = img_placeholder
        #self.x = tf.placeholder(tf.float32,[None,self.feature_dim]) ## important to save real data
        temp_img_placeholder = img_placeholder

        self.encoder_h=[img_placeholder]
        self.decoder_h=[] #out_puts of decoder

        #encoder
        for i in range(len(encoder_shape)):
            print('i=%d'%i)
            with tf.variable_scope('encode_layer{0}'.format(i)):
                ae = new_AE(temp_img_placeholder, encoder_shape[i],sess=self.sess)

                # self.encoder_W.append(ae.getWeigths())
                # self.encoder_b.append(ae.getWeigths())

                temp_img_placeholder = ae.hidden
                self.encoder_h.append(temp_img_placeholder)
                self.encoder_cost.append(ae.cost)
                self.encoder_layers.append(ae)

                self.decoder_h.append()

        # decoder
        for i in range(len(encoder_shape)):
            print('i=%d'%i)
            with tf.variable_scope('decode_layer{0}'.format(i)):
                ae = new_AE(feature_dim, encoder_shape[i])
                self.encoder_W.append(ae.getWeigths())
                self.encoder_b.append(ae.getWeigths())

                self.encoder_h.append(ae.transform(logits))
                logits = ae.transform(logits)

                self.encoder_layers.append(ae)


        # init = tf.global_variables_initializer()
        # self.sess = tf.Session()
        # self.sess.run(init)

    def transform(self,X,layer_idx):
        return self.sess.run(self.encoder_h[layer_idx], feed_dict={self.x: X}) # do not use new new_AE object



        #cost

    # def pretrain(self,X):
    #         ae = self.encoder_layers[i]
    #         if i ==0:
    #             pass
    #         elif:
    #             X = ae.transform(X)
    #         ae.partial_fit(X)


    # def transform(self,idx):
    #     hidden = self.x
    #     for i in range(idx):
    #         hidden= tf.nn.sigmoid(tf.add(tf.matmul(hidden,self.encoder_W[i]),self.encoder_b[i]))
    #
    #     return hidden

if __name__ == '__main__':

    import numpy as np
    import time
    import Config
    from Data import Data
    from Data import Data_Set
    from fully_connected_hsi_classfier_spatial_feature import fill_feed_dict



    # ## data 1------------------------
    dataname = Config.paviaU
    class_num = Config.paviaU_class_num
    # ## data 1------------------------

    # set log dir
    log = './test/'

    with tf.Graph().as_default() as gr:

        pd = Data(dataname)
        data_sets = pd.get_train_valid_test_of_spectral_feature()

        # init data
        train_data = Data_Set([data_sets[0],data_sets[1]])
        valid_data = Data_Set([data_sets[2],data_sets[3]])
        test_data = Data_Set([data_sets[4],data_sets[5]])


        with tf.Session(graph=gr) as sess:
            img_placeholder = tf.placeholder(tf.float32,[None,data_sets[0].shape[1]]) ## important to save real data
            sae = new_SAE(img_placeholder, encoder_shape=[60,60,60,60],sess=sess)

            saver = tf.train.Saver()  # after graph, sometime with variable name ,so so
            init = tf.global_variables_initializer() # order is important,after graph
            sess.run(init)  #after graph

            # tf.summary.scalar('layer0_loss',sae.encoder_cost[0])
            # tf.summary.scalar('layer1_loss',sae.encoder_cost[1])
            # tf.summary.scalar('layer2_loss',sae.encoder_cost[2])
            # tf.summary.scalar('layer3_loss',sae.encoder_cost[3])

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./SAE_graph/',gr)

            for i in range(len(sae.encoder_layers)):
                print('i=%d'%i)
                ae = sae.encoder_layers[i]
                tf.summary.scalar('layer0_loss', sae.encoder_cost[i])

                for step in range(10000):
                    start_time = time.time()
                    feed_dict = fill_feed_dict(train_data, 'input', 'label')
                    X = feed_dict['input']

                    if i>0:
                        X = sae.transform(X,i)
                    else:
                        pass

                    cost = ae.partial_fit(X)
                    duration = time.time() - start_time

                    if step % 1000 == 0:
                        print('loo %d Step %d: loss = %.2f(%.3f sec)' % (i,step, cost, duration))
                        summary_ = sess.run(merged, {ae.x: X})
                        writer.add_summary(summary_,i)