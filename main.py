import os.path
import tensorflow as tf
import helper
import warnings
import glob
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
import project_tests as tests

NUM_CLASSES = 2

IMAGE_SHAPE = (160, 576)

EPOCHS = 20
BATCH_SIZE = 1

LEARNING_RATE = 0.0001 # 0.0009
DROPOUT = 0.75         # 0.5
DATA_DIRECTORY = './data'
RUNS_DIRECTORY = './runs'
TRAINING_DATA_DIRECTORY ='./data/data_road/training'
NUMBER_OF_IMAGES = len(glob.glob('./data/data_road/training/calib/*.*'))
VGG_PATH = './data/vgg'

average_losses = [] # Used for plotting to visualize if our training is going well given parameters
correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
      """
      Load Pretrained VGG Model into TensorFlow.
      :param sess: TensorFlow Session
      :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
      :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
      """
      # TODO: Implement function
      #   Use tf.saved_model.loader.load to load the model and weights
      vgg_tag = 'vgg16'
      vgg_input_tensor_name = 'image_input:0'
      vgg_keep_prob_tensor_name = 'keep_prob:0'
      vgg_layer3_out_tensor_name = 'layer3_out:0'
      vgg_layer4_out_tensor_name = 'layer4_out:0'
      vgg_layer7_out_tensor_name = 'layer7_out:0'

      tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
      graph = tf.get_default_graph()
      image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
      keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
      layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
      layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
      layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

      # the following scaling is based on the suggestion from
      # https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
      # layer3_out_scaled = tf.multiply(layer3_out_raw, 0.0001, name='layer3_out_scaled')
      # layer4_out_scaled = tf.multiply(layer4_out_raw, 0.01, name='layer4_out_scaled')

      return image_input, keep_prob, layer3_out, layer4_out, layer7_out
# tests.test_load_vgg(load_vgg, tf)

def conv_1x1(layer, layer_name):
  """ Return the output of a 1x1 convolution of a layer """
  return tf.layers.conv2d(inputs = layer,
                          filters =  NUM_CLASSES,
                          kernel_size = (1, 1),
                          strides = (1, 1),
                          padding= 'same',
                          kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                          kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                          name = layer_name)
def upsample(layer, k, s, layer_name):
  """ Return the output of transpose convolution given kernel_size k and strides s """
  return tf.layers.conv2d_transpose(inputs = layer,
                                    filters = NUM_CLASSES,
                                    kernel_size = (k, k),
                                    strides = (s, s),
                                    padding = 'same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name = layer_name)
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes = NUM_CLASSES):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  vgg_layerX_out: TF Tensor for VGG Layer X output
  num_classes: Number of classes to classify
  return: The Tensor for the last layer of output
  """

  # Use a shorter variable name for simplicity
  layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

  # Apply a 1x1 convolution to encoder layers
  l3_conv = conv_1x1(layer = layer3, layer_name = "layer3conv1x1")
  l4_conv = conv_1x1(layer = layer4, layer_name = "layer4conv1x1")
  l7_conv = conv_1x1(layer = layer7, layer_name = "layer7conv1x1")

  # Add decoder layers to the network with skip connections and upsampling
  # Note: the kernel size and strides are the same as the example in Udacity Lectures
  #       Semantic Segmentation Scene Understanding Lesson 10-9: FCN-8 - Decoder
  decoderlayer1 = upsample(layer = l7_conv, k = 4, s = 2, layer_name = "decoderlayer1")
  decoderlayer2 = tf.add(decoderlayer1, l4_conv, name = "decoderlayer2")
  decoderlayer3 = upsample(layer = decoderlayer2, k = 4, s = 2, layer_name = "decoderlayer3")
  decoderlayer4 = tf.add(decoderlayer3, l3_conv, name = "decoderlayer4")
  decoderlayer_output = upsample(layer = decoderlayer4, k = 16, s = 8, layer_name = "decoderlayer_output")

  return decoderlayer_output
# tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes = NUM_CLASSES):
  """
  Build the TensorFLow loss and optimizer operations.
  nn_last_layer: TF Tensor of the last layer in the neural network
  correct_label: TF Placeholder for the correct label image
  learning_rate: TF Placeholder for the learning rate
  num_classes: Number of classes to classify
  return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  class_labels = tf.reshape(correct_label, (-1, num_classes))

  # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
  cross_entropy_loss = tf.reduce_mean(cross_entropy)

  # Use AdamOptimizer per suggestion from the walk-through
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

  return logits, train_op, cross_entropy_loss

# tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
  """
  Train neural network and print out the loss during training.
  sess: TF Session
  epochs: Number of epochs
  batch_size: Batch size
  get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  train_op: TF Operation to train the neural network
  cross_entropy_loss: TF Tensor for the amount of loss
  input_image: TF Placeholder for input images
  correct_label: TF Placeholder for label images
  keep_prob: TF Placeholder for dropout keep probability
  learning_rate: TF Placeholder for learning rate
  """

  for epoch in range(EPOCHS):
    losses = []
    i = 0
    for images, labels in get_batches_fn(BATCH_SIZE):
      feed = { input_image: images,
               correct_label: labels,
               keep_prob: DROPOUT,
               learning_rate: LEARNING_RATE }
      _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
      i += 1
      print("epoch: ", i, " partial loss:", loss)
      losses.append(loss)

    average_loss = sum(losses) / len(losses)
    average_losses.append(average_loss)

    print("epoch: ", epoch + 1, " of ", EPOCHS, "average loss: ", average_loss)
# tests.test_train_nn(train_nn)

def run_tests():
  tests.test_layers(layers)
  tests.test_optimize(optimize)
  tests.test_for_kitti_dataset(DATA_DIRECTORY)
  tests.test_train_nn(train_nn)

def run():
  """ Run a train a model and save output images resulting from the test image fed on the trained model """

  # Get vgg model if we can't find it where it should be
  helper.maybe_download_pretrained_vgg(DATA_DIRECTORY)

  # A function to get batches
  get_batches_fn = helper.gen_batch_function(TRAINING_DATA_DIRECTORY, IMAGE_SHAPE)

  with tf.Session() as session:

    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, VGG_PATH)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, NUM_CLASSES)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUM_CLASSES)

    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    # Train the neural network
    train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
             train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

    # Run the model with the test images and save each painted output image (roads painted green)
    helper.save_inference_samples(RUNS_DIRECTORY, DATA_DIRECTORY, session, IMAGE_SHAPE, logits, keep_prob, image_input)

if __name__ == '__main__':
    run_tests()
    run()
    print(average_losses)
    plt.plot(average_losses)
    plt.show()
    plt.savefig("./average_lossses.png")
