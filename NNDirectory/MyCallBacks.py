from keras.callbacks import Callback
import keras.backend as K

class myCallBacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        print('\nLR: {:.6f}\n'.format(lr))