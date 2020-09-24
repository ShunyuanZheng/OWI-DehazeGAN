import tensorflow as tf

#spectral_norm
def _l2normalize(v, eps=1e-12):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)


def max_singular_value(W, u=None, Ip=1):
    if u is None:
        u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False) #1 x ch
    _u = u
    _v = 0
    for _ in range(Ip):
        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
    _v = tf.stop_gradient(_v)
    _u = tf.stop_gradient(_u)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_norm( W, Ip=1):
    u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm",reuse=tf.AUTO_REUSE ):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def layer_norm(x,scope='layer_norm'):
    return tf.contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def group_norm(x,G=32,eps=1e-5,scope='group_norm'):
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C],
                               initializer=tf.constant_initializer(0.0))
        # gamma = tf.reshape(gamma, [1, 1, 1, C])
        # beta = tf.reshape(beta, [1, 1, 1, C])

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x

def gen_conv(batch_input, out_channels ,scope='conv_0'):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    with tf.variable_scope(scope):
        initializer = tf.random_normal_initializer(0, 0.02)
        w=tf.get_variable("kernel", shape=[4, 4, batch_input.get_shape()[-1], out_channels], initializer= initializer,
                          regularizer=None)
        bias=tf.get_variable("bias",[out_channels],initializer=tf.constant_initializer(0.0))
        x=tf.nn.conv2d(input=batch_input,filter=spectral_norm(w),strides=[1,2,2,1],padding='SAME')
        X=tf.nn.bias_add(x,bias)
        return X

def one_conv(batch_input,output_channels):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input,output_channels, kernel_size=1, strides=(1, 1), padding="same",kernel_initializer=initializer)

##resize_conv with spectral_norm
def conv2d_pad(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]], mode=mode)

        return tf.nn.conv2d(x_padded, spectral_norm(weight), strides=[1, strides, strides, 1], padding='VALID', name='conv')

def resize_conv2d(X, input_dim, output_dim, kernel_size, strides):
    """Resizes then applies a convolution.
    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """
    new_h = X.get_shape().as_list()[1]*strides[1]*2
    new_w = X.get_shape().as_list()[2]*strides[2]*2
#    upsized = tf.image.resize_images(X, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    upsized = tf.image.resize_images(X, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    # Now convolve to get the channels to what we want.
#    shape = [kernel_size, kernel_size, input_dim, output_dim]
    h=conv2d_pad(upsized,input_dim,output_dim,kernel_size,strides=strides[1])
    return h

#dilated conv
def dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased=False):
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(x, spectral_norm(w), dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def deconv2d_resize(_input ,num_filter , kernel = 5 , stride=(2,2) , pad = 'SAME' , init_std = 0.05, method =tf.image.ResizeMethod.BILINEAR,
           name = 'deconv2d_resize'):  # use resize before a normal convolution to upsample, avoiding checkerboard effect
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(name ='weight', shape=[kernel,kernel,_input.get_shape()[-1] , num_filter],initializer=tf.truncated_normal_initializer(stddev=init_std , dtype= tf.float32))
        
        width = tf.shape(_input)[2]
        height = tf.shape(_input)[1]
        
        r_width = width*stride[1]*2
        r_height = height*stride[0]*2
        
        resize_img = tf.image.resize_images(_input, [r_height,r_width] , method = method)
        conv = tf.nn.conv2d(resize_img, weight, strides=[ 1, stride[0], stride[1], 1], padding = pad)
        bias = tf.get_variable(name='bias', shape=[num_filter], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, bias)
        output = tf.reshape(output, tf.shape(conv))
        
        return output
