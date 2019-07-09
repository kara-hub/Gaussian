import numpy as np
from keras.models import load_model

model = load_model("seg_net.h5")

test_x = np.load("test_x.npy") # training FFT intensity array [num_sample, 128, 128, 1]
test_y = np.load("test_y.npy") # either training real array or phase array [num_sample, 128, 128, 1]

pred = model.predict(test_x)
np.save("pred_y", pred)


#### Analysis section ####
print(np.sqrt(np.mean((pred-test_y)**2))) # RMSE
# Whatever analysis you want