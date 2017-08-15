import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
from utilsnn import min_max_scale

from modify.au_tmp import AutoEncoder

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX, teY = min_max_scale(trX, teX)

# Autoencoder
autoencoder = AutoEncoder(784, [10, 10, 10, 2], tied_weights=True)

epoch = 2
batchsize = 30
iterations = len(trX) / batchsize
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i, rbm in enumerate(autoencoder.rbm_net):
    for _ in range(epoch):
      for _ in range(batchsize):
        batch_xs, batch_ys = mnist.train.next_batch(batchsize)
        batch_xs = autoencoder.transform_l(batch_xs, i, sess)
        rbm.partial_fit(batch_xs)
    print(rbm.compute_cost(autoencoder.transform_l(trX, i, sess)))
    rbm.save_weights('./rbmw%d.chp' % i)

    # Load RBM weights to Autoencoder
  for i, _ in enumerate(autoencoder.rbm_net):
    autoencoder.load_rbm_weights('./rbmw%d.chp' % i, i, sess)

  # Train Autoencoder
  print('autoencoder')
  for i in range(epoch):
    cost = 0.0
    for j in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batchsize)
        cost += autoencoder.partial_fit(batch_xs, sess)
    print(cost)

  autoencoder.save_weights('./au.chp', sess)
  autoencoder.load_weights('./au.chp', sess)

  fig, ax = plt.subplots()

  print(autoencoder.transform(teX, sess)[:, 0])
  print(autoencoder.transform(teX, sess)[:, 1])

  plt.scatter(
    autoencoder.transform(teX, sess)[:, 0],
    autoencoder.transform(teX, sess)[:, 1],
    alpha=0.5)
  plt.show()

  raw_input("Press Enter to continue...")
  plt.savefig('myfig')
