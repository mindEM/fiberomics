import os
import sys
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, MaxPooling2D, concatenate
#from tensorflow.keras.metrics import binary_crossentropy as bce
from tensorflow.keras.metrics import binary_focal_crossentropy as bce
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow.python.ops.numpy_ops import np_config
import random


class H5Datagenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 path_h5,
                 indexes,
                 batch_size = 4,
                 input_size = (256, 256, 3),
                 shuffle = True):
        self.path_h5 = path_h5
        self.indexes = indexes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(indexes)
        self.num_classes = 3
        
    def __getitem__(self, index):
        new_index = self.indexes[index]
        with h5py.File(self.path_h5, 'r') as f:
            orig = f['X_train'][new_index]
            msk = f['Y_train'][new_index]
            
        return (np.expand_dims(orig, axis = 0), 
                np.expand_dims(msk, axis = 0))
    
    def __len__(self):
        return self.n // self.batch_size
    
    def on_epoch_end(self):
        pass

# Multi-threshold IoU. Outside tf.
np_config.enable_numpy_behavior()

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def diceBCE_loss(y_true, y_pred):
    dice_loss = 1 - dice_coef(y_true, y_pred)
    bce_loss = bce(y_true, y_pred)
    return dice_loss+bce_loss

def calculate_iou(y_true, y_pred):
    iou = []
    y_true = y_true.astype('float')
    y_pred = y_pred.astype('float')
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = (y_pred > t) * 1.
        s_y = y_true + y_pred_
        iou.append(np.sum(s_y == 2) / (np.sum(s_y > 0) + sys.float_info.epsilon))
        
    return np.mean(iou)


def mean_iou(y_true, y_pred):
    iou = tf.py_function(calculate_iou, [y_true, y_pred], tf.float32)
    
    return iou


# Additional bottleneck block for U-net.
def bn_block(in_):
    nf = in_.get_shape().as_list()[3] * 8
    d = 0.1
    kernel = (3, 3)

    # Texture block
    t_b = Conv2D(1, (1, 1),
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(in_)

    t_b = Conv2D(nf, kernel,
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(t_b)

    t_b = Dropout(d)(t_b)

    t_b = Conv2D(nf, kernel,
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(t_b)

    # Color block.
    c_b = Conv2D(nf, (1, 1),
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(in_)

    c_b = Conv2D(nf, kernel,
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(c_b)

    c_b = Dropout(d)(c_b)

    c_b = Conv2D(nf, kernel,
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(c_b)

    return t_b, c_b


def conv_block(in_t, in_c, nf, activation, d):
    mid_ = concatenate([in_c, in_t])
    # mid_ = BatchNormalization()(mid_)
    mid_ = Conv2D(nf, (3, 3),
                  activation=activation,
                  kernel_initializer='he_normal',
                  padding='same')(mid_)
    
    mid_ = Dropout(d)(mid_)
    
    mid_ = Conv2D(nf, (3, 3),
                  activation=activation,
                  kernel_initializer='he_normal',
                  padding='same')(mid_)
    
    return mid_


# Set params.
bs = 8 # batch size
imh, imw = 256, 256
act = 'elu' # activation
don = 0.2 # dropout rate
nn = 32 # initial number of neurons
patience = 50 # patience for early-stopper
out_f_name = 'HCC_double_20221222_focal' # output filename
log_folder = f'./Graph/{out_f_name}' # folder to save tensorboard logs
if not os.path.exists(log_folder):
    os.mkdir(log_folder)
    print("Folder created")

early_stopper = EarlyStopping(monitor='val_loss',
                              patience=patience,
                              verbose=1)

checkpointer = ModelCheckpoint(f'./models/{out_f_name}.h5',
                               verbose=1,
                               save_best_only=True)

tb_callback = TensorBoard(log_dir=f'./Graph/{out_f_name}/',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)


# Build model.
s = Input((imh, imw, 3))

t1, c1 = bn_block(s)
m1 = conv_block(c1, t1, nn, act, don)
p1 = MaxPooling2D((2, 2))(m1)

t2, c2 = bn_block(p1)
m2 = conv_block(c2, t2, nn, act, don)
p2 = MaxPooling2D((2, 2))(m2)

t3, c3 = bn_block(p2)
m3 = conv_block(c3, t3, nn, act, don)
p3 = MaxPooling2D((2, 2))(m3)

t4, c4 = bn_block(p3)
m4 = conv_block(c4, t4, nn, act, don)
p4 = MaxPooling2D((2, 2))(m4)

t5, c5 = bn_block(p4)
m5 = conv_block(c5, t5, nn, act, don)
p5 = MaxPooling2D((2, 2))(m5)

t6, c6 = bn_block(p5)
m6 = conv_block(c6, t6, nn, act, don)

u1 = Conv2DTranspose(nn, (2, 2),
                     strides=(2, 2),
                     padding='same')(m6)
m7 = conv_block(u1, m5, nn, act, don)

u2 = Conv2DTranspose(nn, (2, 2),
                     strides=(2, 2),
                     padding='same')(m7)
m8 = conv_block(u2, m4, nn, act, don)

u3 = Conv2DTranspose(nn, (2, 2),
                     strides=(2, 2),
                     padding='same')(m8)
m9 = conv_block(u3, m3, nn, act, don)

u4 = Conv2DTranspose(nn, (2, 2),
                     strides=(2, 2),
                     padding='same')(m9)
m10 = conv_block(u4, m2, nn, act, don)

u5 = Conv2DTranspose(nn, (2, 2),
                     strides=(2, 2),
                     padding='same')(m10)
m11 = conv_block(u5, m1, nn, act, don)

o = Conv2D(3, (1, 1), activation='sigmoid')(m11)

model = Model(inputs=[s], outputs=[o])
model.compile(optimizer='adam',
              loss=diceBCE_loss,
              metrics=[mean_iou])

model.summary()

# Load training data to memory.
path_h5 = '/path/to/your/training/dataset.h5'
with h5py.File(path_h5,'r')as f:
    train_indexes = list(range(f['X_train'].shape[0]))

valid_indexes = random.sample(train_indexes, int(0.2 * len(train_indexes)))

for i in valid_indexes:
    train_indexes.remove(i)

traingen = H5Datagenerator(path_h5, train_indexes, batch_size = bs)
validgen = H5Datagenerator(path_h5, valid_indexes, batch_size = bs)

# Train the model.
model.fit(traingen,
          validation_data = validgen,
          batch_size=bs, epochs=100000,
          callbacks=[early_stopper,
                     checkpointer,
                     tb_callback])
