import tensorflow as tf
from new_SAE import new_SAE
import Config
from Data import Data
from Data import Data_Set
import time
import os.path
from Data import next_feed_dict


def run_trainning(data_sets,final_ckpt_dir,pre_train_ckpt_dir,transfer_function,pre_need_train=False,final_need_train=False):


    pre_train_graph_path = os.path.join(pre_train_ckpt_dir, 'SAE_graph')
    pre_train_ckpt_path = os.path.join(pre_train_ckpt_dir, 'ckpt')


    final_train_graph_path = os.path.join(final_ckpt_dir, 'final_graph')
    final_train_ckpt_path = os.path.join(final_ckpt_dir, 'ckpt')

    #transfer_function = transfer_function



    with tf.Graph().as_default() as gad:

        data_sets =data_sets

        # init class
        train_data = Data_Set([data_sets[0],data_sets[1]])
        valid_data = Data_Set([data_sets[2],data_sets[3]])
        test_data = Data_Set([data_sets[4],data_sets[5]])

        writer = tf.summary.FileWriter(final_train_graph_path, gad)

        with tf.Session(graph = gad) as sess:

            input_dim = [train_data.feature_dim,train_data.labels_dim]
            epoch_size = data_sets[0].shape[0]

            # int
            sae = new_SAE(input_dim, encoder_shape=Config.encoder_layers,
                          data_sets_shape=[train_data._num_examples,valid_data._num_examples,test_data._num_examples],
                          transfer_function=transfer_function)

            pre_trian_saver = tf.train.Saver()  # after graph, sometime with variable name ,so so
            fianl_saver = tf.train.Saver()  # after graph, sometime with variable name ,so so


            init = tf.global_variables_initializer() # order is important,after graph
            sess.run(init)  #after graph

            # if pretrain ready
            if pre_need_train == True:

                sae.pre_train(train_data,sess,pre_train_graph_path)
                pre_trian_saver.save(sess, os.path.join(pre_train_ckpt_path, 'pre_train_model.ckpt'))

            else:
                ckpt = tf.train.get_checkpoint_state(pre_train_ckpt_path)
                pre_trian_saver.restore(sess, ckpt.model_checkpoint_path)

            # train real model
            if final_need_train == True:
                for step in range(int(Config.epoch_final_train_times * epoch_size / Config.batch_size)):
                    start_time = time.time()
                    feed_dict = next_feed_dict(train_data, sae.x, sae.y)

                    cost,_= sae.train_final_model(feed_dict,sess)

                    duration = time.time() - start_time

                    if step % 1000 == 0:
                        print('final model train : Step %d: loss = %.5f(%.3f sec)' % (step, cost, duration))
                        summary_loss = sess.run(sae.final_merged, feed_dict=feed_dict)
                        writer.add_summary(summary_loss,step)
                        fianl_saver.save(sess, os.path.join(final_train_ckpt_path, 'final_model.ckpt'))


                        precision1 = sess.run(sae.one_batch_precsion,
                                             feed_dict=feed_dict)
                        print(' Train data evaluation, one batch data precision=%10f' % precision1)

                        precision2 = sess.run(sae.all_train_precision,
                                             feed_dict={sae.x: train_data._images,
                                                        sae.y: train_data._labels})
                        print(' Train data evaluation, all train data precision=%10f' % precision2)

                        summary_p1 = sess.run(sae.precision_train_merged,
                                             feed_dict={sae.x: train_data._images,
                                                        sae.y: train_data._labels})
                        writer.add_summary(summary_p1, step)

                    if step % 1000 == 0 or (step + 1) == Config.max_steps:
                        print(' Valid data evaluation')
                        summary_p2 = sess.run(sae.precision_valid_merged,
                                             feed_dict={sae.x: valid_data._images,
                                                        sae.y: valid_data._labels})
                        writer.add_summary(summary_p2, step)

                    if step % 1000 == 0 or (step + 1) == Config.max_steps:
                        print(' Test data evaluation')
                        summary_p3 = sess.run(sae.precision_test_merged,
                                             feed_dict={sae.x: test_data._images,
                                                        sae.y: test_data._labels})
                        writer.add_summary(summary_p3, step)
            else:

                ckpt = tf.train.get_checkpoint_state(final_train_ckpt_path)
                fianl_saver.restore(sess, ckpt.model_checkpoint_path)


                for step in range(int(Config.epoch_final_train_times * epoch_size / Config.batch_size)):
                    start_time = time.time()
                    feed_dict = next_feed_dict(train_data, sae.x, sae.y)

                    cost,_= sae.train_final_model(feed_dict,sess)

                    duration = time.time() - start_time

                    if step % 1000 == 0:
                        print('final model train : Step %d: loss = %.5f(%.3f sec)' % (step, cost, duration))
                        summary_loss = sess.run(sae.final_merged, feed_dict=feed_dict)
                        writer.add_summary(summary_loss,step)
                        fianl_saver.save(sess, os.path.join(final_train_ckpt_path, 'final_model.ckpt'))


                        precision1 = sess.run(sae.one_batch_precsion,
                                             feed_dict=feed_dict)
                        print(' Train data evaluation, one batch data precision=%10f' % precision1)

                        precision2 = sess.run(sae.all_train_precision,
                                             feed_dict={sae.x: train_data._images,
                                                        sae.y: train_data._labels})
                        print(' Train data evaluation, all train data precision=%10f' % precision2)

                        summary_p1 = sess.run(sae.precision_train_merged,
                                             feed_dict={sae.x: train_data._images,
                                                        sae.y: train_data._labels})
                        writer.add_summary(summary_p1, step)

                    if step % 1000 == 0 or (step + 1) == Config.max_steps:
                        print(' Valid data evaluation')
                        summary_p2 = sess.run(sae.precision_valid_merged,
                                             feed_dict={sae.x: valid_data._images,
                                                        sae.y: valid_data._labels})
                        writer.add_summary(summary_p2, step)

                    if step % 1000 == 0 or (step + 1) == Config.max_steps:
                        print(' Test data evaluation')
                        summary_p3 = sess.run(sae.precision_test_merged,
                                             feed_dict={sae.x: test_data._images,
                                                        sae.y: test_data._labels})
                        writer.add_summary(summary_p3, step)
if __name__ == "__main__":

    dataname = Config.paviaU

    mix_log = '/' + dataname + '/' + 'mix'
    pre_need_train = True
    final_need_train = True

    final_model_dir = Config.final_model_dir + mix_log
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    pre_train_model_dir = Config.pretrain_model_dir + mix_log
    if not os.path.exists(pre_train_model_dir):
        os.makedirs(pre_train_model_dir)

    pd = Data(data_name=dataname)
    data_sets = pd.get_train_valid_test_of_mix_feature()

    run_trainning(data_sets, final_model_dir, pre_train_model_dir, pre_need_train=pre_need_train, final_need_train=final_need_train)



