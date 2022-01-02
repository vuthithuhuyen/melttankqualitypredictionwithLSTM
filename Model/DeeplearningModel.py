import pickle

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def SaveDictToFile(_dict, _fileName):
    pickle_out = open(_fileName, "wb")
    pickle.dump(_dict, pickle_out)
    pickle_out.close()


def LoadFileToDict(_fileName):
    pickle_in = open(_fileName, "rb")
    myDict = pickle.load(pickle_in)
    return myDict


def prepare_lstm_model(input_shape, len_outputs_vector, _n_out):
    model = Sequential()

    print(f'Input_shape for model: {input_shape}')
    model.add(LSTM(50, input_shape=input_shape))
    model.add(RepeatVector(_n_out * 1))  # Important! For connecting to time distributed layer

    # model.add(LSTM(150, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(len_outputs_vector, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def prepare_mlp_model(dim, name_to_int_dict):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model.add(Dense(len(name_to_int_dict), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def GetReverseYValue(y_predict, name_to_int_dict):
    key = numpy.argmax(y_predict)
    predict = name_to_int_dict[key]
    return predict


def convert_back_y_predicted(y_predict, name_to_int_dict):
    predict_list = []
    for val in y_predict:
        key = numpy.argmax(val)

        current_predict_vector = [0 for _ in range(len(name_to_int_dict))]
        current_predict_vector[key] = 1
        predict_list.append(current_predict_vector)
    y_predict = numpy.asarray(predict_list)
    return y_predict


def EvaluateResult(y_predict, y_original, name_to_int_dict):
    predict_list = []
    y_original = y_original.reshape(-1, len(name_to_int_dict))
    for val in y_predict:
        key = numpy.argmax(val)

        current_predict_vector = [0 for _ in range(len(name_to_int_dict))]
        current_predict_vector[key] = 1
        predict_list.append(current_predict_vector)
    y_predict = numpy.asarray(predict_list)

    count = 0
    for row in range(len(y_predict)):
        compare = y_predict[row] == y_original[row]
        if all(x == True for x in compare):
            count += 1
    presicion = count / len(y_predict)
    return presicion
