from typing import List
import keras
from keras import Sequential
from keras.layers import Dense, Dropout


class NNparams:
    hidden = List[int]
    activation = None
    dropout = List[int]
    train_metric = None
    batch_size = None
    l1 = float
    l2 = float
    optimizer = None
    _model = None
    loss = None
    input_dim = int
    kernel_initializer = None
    bias_initializer = None

    def _buildNNModel(self):
        model = Sequential()
        for i in range(0, len(self.hidden)):
            if i == 0:
                model.add(Dense(self.hidden[i], input_dim=self.input_dim,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                activation=self.activation,
                                activity_regularizer=keras.regularizers.l1_l2(self.l1, self.l2)
                                ))
                #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
                model.add(Dropout(self.dropout[i]))
            if i != 0:
                model.add(Dense(self.hidden[i],
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                activation=self.activation,
                                activity_regularizer=keras.regularizers.l1_l2(self.l1, self.l2)
                                ))
                #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.01))
                model.add(Dropout(self.dropout[i]))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.train_metric)
        return model

    def get_nn_model(self):
        return self._model

    def __init__(self, hidden, dropout, optimizer, l1reg, l2reg, activation, input_dim,
                 loss, train_metric, batch_size, kernel_init, bias_init):
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.l1 = l1reg
        self.l2 = l2reg
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.loss = loss
        self.kernel_initializer = kernel_init
        self.bias_initializer = bias_init
        self.train_metric = train_metric
        self._model = self._buildNNModel()
