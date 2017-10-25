import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# NUM_CLASSES = 2
# IMAGE_SHAPE = (160, 576)

EPOCHS = 20
BATCH_SIZE = 1

LEARNING_RATE = 0.0009 # 0.0001
DROPOUT = 0.5 # 0.75
average_losses = []             # for plotting the average_losses

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

      tf.save_model.loader.load(sess, [vgg_tag], vgg_tag)
      graph = tf.get_default_graph()
      image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
      keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
      layer3_out_raw = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
      layer4_out_raw = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
      layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

      # the following scaling is based on the suggestion from
      # https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
      layer3_out_scaled = tf.multiply(layer3_out_raw, 0.0001, name='layer3_out_scaled')
      layer4_out_scaled = tf.multiply(layer4_out_raw, 0.01, name='layer4_out_scaled')

      return image_input, keep_prob, layer3_out_scaled, layer4_out_scaled, layer7_out
tests.test_load_vgg(load_vgg, tf)

def conv_1x1(layer, layer_name, num_classes):
    """
    'return the 1x1 convolution of a layer
    """
    return tf.layers.conv2d(inputs=layer,
                            num_classes,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding= 'same',
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                            name=layer_name)
def upsample(layer, kernel, stride, layer_name, num_classes):
    """
    return the convolution transpose of the layer given kernel and stride.
    """
    return tf.layers.conv2d_traspose(inputs=layer,
                                     filters=num_classes,
                                     kernel_size=(kernel, kernel),
                                     strides=(stride, stride),
                                     padding='same',
                                     kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name=layer_name)
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    l7_conv = conv_1x1(vgg_layer7_out, 'l7_conv', num_classes)
    l4_conv = conv_1x1(vgg_layer4_out, 'l4_conv', num_classes)
    l3_conv = conv_1x1(vgg_layer3_out, 'l3_conv', num_classes)

    l7_conv_upsample = upsample(l7_conv, 4, 2, 'l7_conv_upsample', num_classes)
    # add skip l4_conv
    l4_skip_added = tf.add(l7_conv_upsample, l4_conv)
    l4_skip_added_upsample = upsample(l4_skip_added, 4, 2, "l4_skip_added_upsample", num_classes)
    l3_skip_added = tf.add(l4_skip_added_upsample, l3_conv)
    output = upsample(l3_skip_added, 16, 8, 'output', num_classes)

    # example of print out the dimension for debug
    # tf.Print(output, [tf.shape(output)[1:3]])
    return output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # make logits a 2D tensor where each row represents a pxel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                label=correct_label))
    # training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    // sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    losses = []
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch))
        for image, label in get_batches_fn(batch_size):
            # performe training
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: LEARNING_RATE,
                                          learning_rate: DROPOUT})
            losses.append(loss)  # record loss for plotting
        #end of for image, label
        average_loss = sum(losses)/len(losses)
        average_losses = append(average_losses)

        print("Loss: = {:.3f}".format(loss))
    #end of for epoch
    print()
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # TF placeholders
        correct_label = tf.placeholde(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        model_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes);

        # TODO: Train NN using the train_nn function

        # initialize variables
        sess.run(tf.gloabl_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video



if __name__ == '__main__':
    run()
    print(average_losses)
