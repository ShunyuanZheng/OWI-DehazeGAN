"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf

import layers

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

# The number of color channels per image.
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64


def get_outputs(inputs, network="tensorflow", skip=False):
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model") as scope:

        if network == "pytorch":
            current_discriminator = discriminator
            current_generator = build_generator_resnet_9blocks
        elif network == "tensorflow":
            current_discriminator = discriminator_tf
            current_generator = build_dehaze_generator_resnet_9blocks_tf
        else:
            raise ValueError(
                'network must be either pytorch or tensorflow'
            )

        prob_real_a_is_real = current_discriminator(images_a, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, "d_B")

        fake_images_b = current_generator(images_a, name="g_A", skip=skip)
        fake_images_a = current_generator(images_b, name="g_B", skip=skip)

        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(fake_images_a, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, "d_B")

        cycle_images_a = current_generator(fake_images_b, "g_B", skip=skip)
        cycle_images_b = current_generator(fake_images_a, "g_A", skip=skip)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
    }


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_dehaze_generator_resnet_9blocks_tf(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1", relufactor=0.2)#256*256*32
        
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)#128*128*64

        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)#64*64*128

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)#64*64*128
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        with tf.variable_scope("resize_conv1"):
            o_c4 = tf.concat([o_r9,o_c3],3)
            o_c4 = layers.instance_norm(o_c4)
        #o_c4 = layers.resize_conv2d(o_c4, 256, ngf*2, 2, 1)
            o_c4 = layers.deconv2d_resize(o_c4, ngf*2, kernel=ks, stride=(2,2), name='deconv_1')
            o_c4 = layers.lrelu(o_c4)
      #  o_c4 = layers.general_deconv2d(
        #    o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
        #    "SAME", "c4")
        with tf.variable_scope("resize_conv2"):
            o_c5 = tf.concat([o_c2, o_c4],3)
            o_c5 = layers.instance_norm(o_c5)
       # o_c5 = layers.resize_conv2d(o_c5, 128, ngf, 2, 1)
            o_c5 = layers.deconv2d_resize(o_c5, ngf, kernel=ks, stride=(2,2), name='deconv_2')
            o_c5 = layers.lrelu(o_c5)
      #  o_c5 = layers.general_deconv2d(
       #     o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
        #    "SAME", "c5")
        
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen

def dehaze_generator_add_tf(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1", relufactor=0.2)#256*256*32
        
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)#128*128*64

        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)#64*64*128

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)#64*64*128
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        with tf.variable_scope("resize_conv1"):
            #o_c4 = tf.concat([o_r9,o_c3],3)
            o_c4 = o_c3 + o_r9
            o_c4 = layers.instance_norm(o_c4)
            o_c4 = layers.deconv2d_resize(o_c4, ngf*2, kernel=ks, stride=(2,2), name='deconv_1')
            o_c4 = layers.lrelu(o_c4)
      #  o_c4 = layers.general_deconv2d(
        #    o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
        #    "SAME", "c4")
        with tf.variable_scope("resize_conv2"):
            #o_c5 = tf.concat([o_c2, o_c4],3)
            o_c5 = o_c2 + o_c4
            o_c5 = layers.instance_norm(o_c5)
            o_c5 = layers.deconv2d_resize(o_c5, ngf, kernel=ks, stride=(2,2), name='deconv_2')
            o_c5 = layers.lrelu(o_c5)
      #  o_c5 = layers.general_deconv2d(
       #     o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
        #    "SAME", "c5")
        
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen

def dehaze_resize_with_deconv(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1", relufactor=0.2)#256*256*32
        
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)#128*128*64

        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)#64*64*128

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)#64*64*128
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        with tf.variable_scope("resize_conv1"):
            o_c4_0 = tf.concat([o_r9,o_c3],3)
            o_c4_1 = layers.instance_norm(o_c4_0)
            o_c4_1 = layers.deconv2d_resize(o_c4_1, ngf*4, kernel=ks, stride=(2,2), name='deconv_1')
            o_c4_1 = layers.lrelu(o_c4_1)
            o_c4_2 = layers.general_deconv2d(o_c4_0, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,"SAME", "c4")
            o_c4_3 = tf.concat([o_c4_1,o_c4_2],3)
            o_c4_4 = layers.one_conv(o_c4_3, 64)
            o_c4_4 = layers.lrelu(o_c4_4)
            

        with tf.variable_scope("resize_conv2"):
            o_c5_0 = tf.concat([o_c2, o_c4_4],3)
            o_c5 = layers.instance_norm(o_c5_0)
            o_c5_1 = layers.deconv2d_resize(o_c5, ngf*2, kernel=ks, stride=(2,2), name='deconv_2')
            o_c5_1 = layers.lrelu(o_c5_1)

            o_c5_2 = layers.general_deconv2d(o_c5_0, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,"SAME", "c5")
            o_c5_3 = tf.concat([o_c5_1, o_c5_2],3)
            o_c5_4 = layers.one_conv(o_c5_3, 32)
            o_c5_4 = layers.lrelu(o_c5_4)

        with tf.variable_scope("Output_layer"):
          #  o_c6_0 = tf.concat([o_c5_1, o_c5_2],3)
        
            o_c6 = layers.general_conv2d(o_c5_4, IMG_CHANNELS, f, f, 1, 1,0.02, "SAME", "c6",do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def dehaze_generator(generator_inputs, name="generator", skip=False):

    # Encoder
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]

    with tf.variable_scope("process_conv1",reuse=tf.AUTO_REUSE):
        process1 = layers.conv2d_pad(generator_inputs, 3, ndf, 5, 1)
        process1 = layers.lrelu(process1, 0.2)
    with tf.variable_scope("process_conv2",reuse=tf.AUTO_REUSE):
        process2 = layers.conv2d_pad(process1, ndf, ndf, 3, 1)
        process2 = layers.lrelu(process2, 0.2)


    with tf.variable_scope("encoder_1",reuse=tf.AUTO_REUSE):
        rectified = layers.lrelu(process2, 0.2)
        encoder1 = layers.gen_conv(rectified, ndf, scope='conv_0')

        encoder1 = layers.instance_norm(encoder1)

    with tf.variable_scope("encoder_2",reuse=tf.AUTO_REUSE):
        rectified = layers.lrelu(encoder1, 0.2)
        encoder2 = layers.gen_conv(rectified, ndf * 2, scope='conv_1')
     
        encoder2 = layers.instance_norm(encoder2)

    with tf.variable_scope("encoder_3",reuse=tf.AUTO_REUSE):
        rectified = layers.lrelu(encoder2, 0.2)
        encoder3 = layers.gen_conv(rectified, ndf * 4, scope='conv_2')
      
        encoder3 = layers.instance_norm(encoder3)

    with tf.variable_scope("encoder_4", reuse=tf.AUTO_REUSE):
        rectified = layers.lrelu(encoder3, 0.2)
        encoder4 = layers.gen_conv(rectified, ndf * 8, scope='conv_3')
    
        encoder4 = layers.instance_norm(encoder4)

    padding = "REFLECT"
    # Residual_Block
    o_r1 = build_resnet_block(encoder4, ndf * 8, "r1", padding)
    o_r2 = build_resnet_block(o_r1, ndf * 8, "r2", padding)
    o_r3 = build_resnet_block(o_r2, ndf * 8, "r3", padding)
    o_r4 = build_resnet_block(o_r3, ndf * 8, "r4", padding)
    o_r5 = build_resnet_block(o_r4, ndf * 8, "r5", padding)
    # o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
    # o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
    # o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
    # o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

    # decoder
    with tf.variable_scope("Upsample_1",reuse=tf.AUTO_REUSE):
        rectified = tf.concat([encoder4, o_r5], 3)

        rectified = layers.instance_norm(rectified)
        upsample1 = layers.lrelu(rectified, 0.2)
        upsample1 = layers.resize_conv2d(upsample1, 1024, 256, 3, [1, 1, 1, 1])
        upsample1 = tf.nn.dropout(upsample1, keep_prob=0.5)
  

    with tf.variable_scope("Upsample_2",reuse=tf.AUTO_REUSE):
        rectified = tf.concat([encoder3, upsample1], 3)
        rectified = layers.instance_norm(rectified)
        upsample2 = layers.lrelu(rectified, 0.2)
        upsample2 = layers.resize_conv2d(upsample2, 512, 128, 3, [1, 1, 1, 1])
        upsample2 = tf.nn.dropout(upsample2, keep_prob=0.5)
     

    with tf.variable_scope("Upsample_3",reuse=tf.AUTO_REUSE):
        rectified = tf.concat([encoder2, upsample2], 3)
        rectified = layers.instance_norm(rectified)
        upsample3 = layers.lrelu(rectified, 0.2)
        upsample3 = layers.resize_conv2d(upsample3, 256, 64, 3, [1, 1, 1, 1])
      

    with tf.variable_scope("Upsample_4",reuse=tf.AUTO_REUSE):
        rectified = tf.concat([encoder1, upsample3], 3)
        rectified = layers.instance_norm(rectified)
        upsample4 = layers.lrelu(rectified, 0.2)
        upsample4 = layers.resize_conv2d(upsample4, 128, 3, 3, [1, 1, 1, 1])
        Upsample_4 = tf.tanh(upsample4)
    

    with tf.variable_scope("Multi-scale_Refine_1", reuse=tf.AUTO_REUSE):
        rectified = tf.concat([Upsample_4, process2], 3)
        rectified = layers.instance_norm(rectified)
        conv3_1 = layers.conv2d_pad(rectified, 67, 64, 3, 1)
        conv3_1 = layers.lrelu(conv3_1, 0.2)
        conv5_1 = layers.conv2d_pad(rectified, 67, 64, 5, 1)
        conv5_1 = layers.lrelu(conv5_1, 0.2)

    with tf.variable_scope("Multi-scale_Refine_2", reuse=tf.AUTO_REUSE):
        conv_1 = tf.concat([conv3_1, conv5_1], 3)
        conv1_1 = layers.instance_norm(conv_1)
        conv3_2 = layers.conv2d_pad(conv1_1, 128, 128, 3, 1)
        conv3_2 = layers.lrelu(conv3_2, 0.2)
        conv5_2 = layers.conv2d_pad(conv1_1, 128, 128, 5, 1)
        conv5_2 = layers.lrelu(conv5_2, 0.2)

    with tf.variable_scope("Multi-scale_Refine_3", reuse=tf.AUTO_REUSE):
        conv_2 = tf.concat([conv3_2, conv5_2], 3)
        conv2_2 = layers.instance_norm(conv_2)
        conv_out = layers.lrelu(layers.one_conv(conv2_2, 32), 0.2)
        conv_out = layers.conv2d_pad(conv_out, 32, 3, 3, 1)
        mul_out = tf.tanh(conv_out)

    if skip is True:
        out_gen = tf.nn.tanh(generator_inputs + mul_out, "t1")
    else:
        out_gen = tf.nn.tanh(mul_out, "t1")

    return out_gen


def build_generator_resnet_9blocks(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)

        o_c4 = layers.general_deconv2d(
            o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")
        o_c5 = layers.general_deconv2d(
            o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "c5")
        o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c6",
                                     do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def discriminator_tf(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        o_c1 = layers.general_conv2d(inputdisc, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02,
            "SAME", "c5", do_norm=False, do_relu=False
        )

        return o_c5


def discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2,
                                     0.02, "VALID", "c1", do_norm=False,
                                     relufactor=0.2)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "VALID", "c2", relufactor=0.2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "VALID", "c3", relufactor=0.2)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "VALID", "c4", relufactor=0.2)

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(
            pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5",
            do_norm=False, do_relu=False)

        return o_c5


def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 3])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5
