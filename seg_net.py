from keras.models import Sequential
from keras.layers import Activation, Input, Concatenate, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def seg_net(num_encode_channels=[[32, 32], [64, 64], [128, 128], [256, 256]],
			encode_kernel_sizes=[[3, 3], [3, 3], [3, 3], [3,3]],
			num_decode_channels=[[256, 256], [128, 128], [64, 64], [32, 32]],
			decode_kernel_sizes=[[3, 3], [3, 3], [3, 3], [3,3], [3,3]],
			input_shape=(128, 128, 1),
			reg=l2(0)):

	"""
	Keras implementation of SegNet.
	num_encode_channels: each block is a list, after each block, there is a maxpooling layer
	encode_kernel_sizes: the size of kernel of each layer, must be the same shape as num_encode_channels
	num_decode_channels: each block is a list, before each block, there is a upsampling layer
	decode_kernel_sizes: the size of kernel of each layer, must be the same shape as num_decode_channels
	input_shape: the input shape for each image, (width, height, #channels)
	"""

	model = Sequential()
	for i in range(len(num_encode_channels)):
		num_channel = num_encode_channels[i]
		kernel_size = encode_kernel_sizes[i]
		for j in range(len(num_channels)):
			n = num_channel[j]
			k = kernel_size[j]
			if i == 0 and j == 0:
				model.add(Convolution2D(n, k, border_mode='same', input_shape=input_shape, W_regularizer=reg))
			else:
				model.add(Convolution2D(n, k, border_mode='same'))
		    model.add(BatchNormalization())
		    model.add(Activation('relu'))
		model.add(MaxPooling2D())
	for i in range(len(num_decode_channels)):
		model.add(UpSampling2D())
		num_channel = num_decode_channels[i]
		kernel_size = decode_kernel_sizes[i]
		for j in range(len(num_channels)):
			n = num_channel[j]
			k = kernel_size[j]
			model.add(Convolution2D(n, k, border_mode='same', W_regularizer=reg))
		    model.add(BatchNormalization())
		    model.add(Activation('relu'))
    model.add(Convolution2D(1, 3, border_mode='same'))
    return model


def new_net(sizes=[16, 32, 64, 128],
	        channels=[[[64, 64, 64], [128, 128, 128]], # first segnet
	                  [[32, 32, 32], [64, 64, 64], [128, 128, 128]],
	                  [[16, 16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]],
	                  [[8, 8, 8, 8], [16, 16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]]],
			kernels=[[[3, 3, 3], [3, 3, 3]], # first segnet
					[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
					[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
					[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]],
			input_shape=(128, 128, 1)):

	"""
	Keras implementation of SegNet.
	sizes: the sizes the centered FFT
	seg_nets: the same length as "sizes", each of the list is a segnet hyperparameter
	kernel: the same length as "sizes", each of the list is a segnet hyperparameter
	input_shape: the input shape for each image, (width, height, #channels)
	"""

	inp = Input(input_shape)
	all_nets = []
	for i in range(len(sizes)):
		size = sizes[i]
		net_input = Lambda(x: inp[:,inp_shape[0]//2-size//2:inp_shape[0]//2+size//2, inp_shape[1]//2-size//2:inp_shape[1]//2+size//2])(inp)
		channel = channels[i]
		decode_channel = list(channel)
		decode_channel.reverse()
		kernel = kernels[i]
		decode_kernel = list(kernel)
		decode_kernel.reverse()
		net_arch = seg_net(channel, kernel, decode_channel, decode_kernel, (size, size, input_shape[-1]))
		all_nets.append(net_arch(net_input))
	net_output = all_nets[0]
	for i in range(1, len(all_nets)):
		net_output = UpSampling2D()(net_output)
		net_output = Concatenate()([net_output, all_nets[i]])
		for j in [64, 32, 1]:
			net_output = Convolution2D(j, 3, border_mode='same')(net_output)
	y = Convolution2D(1, 3, border_mode='same')(net_output)
    return Model(inp, y)



