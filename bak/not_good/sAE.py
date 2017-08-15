
import tensorflow as tf
import math
import Config


def inference(images,encoder_layers_shape):

    layer_units1 = int(images.shape[1])
    print(layer_units1)
    encoder_layers_shape = encoder_layers_shape

    # initial encoder
    for i in range(len(encoder_layers_shape)):

        layer_units2= encoder_layers_shape[i]
        # set layer scope for each encoder layer
        with tf.name_scope('encoder_layer{0}'.format(i)):

            weights = tf.Variable(
                tf.truncated_normal([layer_units1,layer_units2],
                                    stddev=1.0/math.sqrt(float(layer_units1))),
                name='weights{0}'.format(i))

            biases =tf.Variable(tf.zeros([layer_units2]),name='biases{0}'.format(i))


            images = tf.nn.relu(tf.matmul(images,weights)+biases)

            layer_units1 = encoder_layers_shape[i]

    # encoder output as decoder input
    de_layer_units1 = layer_units1
    # initial dencoder
    for i in range(len(encoder_layers_shape)):

        layer_units2 = encoder_layers_shape[-i]
        # set layer scope for each decoder layer
        with tf.name_scope('decoder_layer{0}'.format(i)):
            weights = tf.Variable(
                tf.truncated_normal([de_layer_units1,layer_units2],
                                    stddev=1.0/math.sqrt(float(de_layer_units1))),
                name='weights{0}'.format(i))

            biases =tf.Variable(tf.zeros([layer_units2]),name='biases{0}'.format(i))

            images = tf.nn.relu(tf.matmul(images,weights)+biases)
            de_layer_units1 = encoder_layers_shape[-i]

    #decoder output as output layer input
    output_layer_input = layer_units1
    # initial output layer
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([output_layer_input, Config.NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(output_layer_input))),
            name='weights_out')

        biases = tf.Variable(tf.zeros([Config.NUM_CLASSES]), name='biases_out')

        logits = tf.matmul(images, weights) + biases

    return logits


def loss(logits,labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,logits=logits,name='xentropy')

    return tf.reduce_mean(cross_entropy,name='xentropy_mean')


def training(loss,learning_rate):

    # Greate a summarizer to track the loss over time in TensorBoard
    tf.summary.scalar('loss',loss)

    # Create a optimization method
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Set the steps
    global_step = tf.Variable(0,name='global_step',trainable=False)

    # Set the trainning objection to minimize loss
    train_op = optimizer.minimize(loss,global_step=global_step)

    return train_op


def pre_train():

    return


def evaluation(logits,labels,k=1):

    correct = tf.nn.in_top_k(logits,labels,k) # if the maxmum k value could match labels, return True


    return tf.reduce_sum(tf.cast(correct,tf.int32))



if __name__ == '__main__':

    import numpy as np

    labels = np.array([1,2,3])
    print(labels.shape)
    logits = np.array([[0.5,0.5,0.2,0.3],[0.5,0.5,0.2,0.3],[0.5,0.5,0.2,0.3]]) # from 0-3, 4 rank tensor in one prediction

    #logits = [1,3,3]

    sess = tf.Session()

    eval_correct = evaluation(logits, labels,k=2)

    print(sess.run(eval_correct))