import numpy as np
from seg_net import seg_net, new_net

model = seg_net() # comment this line and uncomment next line when you use new network
#model = new_net()

train_x = np.load("train_x.npy") # training FFT intensity array, [num_sample, 128, 128, 1]
train_y = np.load("train_y.npy") # either training real array or phase array, [num_sample, 128, 128, 1]

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
model.fit(train_x, train_y, batch_size=128, nb_epoch=10, shuffle=True)
model.save("seg_net.h5") # change the name as you want
