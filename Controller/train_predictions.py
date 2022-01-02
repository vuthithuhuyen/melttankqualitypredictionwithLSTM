import gc
from idlelib import history
import numpy as np

import numpy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from Helper.DataPreprocessing import ScaleData, series_to_supervised, one_hot_decode, vector_to_label

# Read raw .csv data to X-Input and y-output
from Model import GlobalVariables
from Model.DeeplearningModel import prepare_mlp_model, EvaluateResult, prepare_lstm_model, convert_back_y_predicted
import pandas as pd
from datetime import datetime, time

# Prepare Data
data = pd.read_csv(GlobalVariables.raw_data_file)

# 200k ~ 300k
data = data[200000:600000]
# data = data[200000:210000]

def cleanup_memory():
    # backend.clear_session()
    gc.collect()

def train_predict(_data, _cols, _n_inputs, _n_outputs, _epochs):
    inputfeatures = _cols[:-1]
    n_inputfeatures = len(inputfeatures)
    _data = _data[_cols]
    X, y = series_to_supervised(_data, _n_inputs, n_out=_n_outputs, dropnan=True)
    # Scale data
    X, y, name_to_int, int_to_name = ScaleData(X, y)

    # split train, test
    split = train_test_split(X, y, test_size=GlobalVariables.test_size)
    (train_X, test_X, train_y, test_y) = split

    train_X = train_X.reshape(-1, _n_inputs, n_inputfeatures)
    test_X = test_X.reshape(-1, _n_inputs, n_inputfeatures)

    train_y = train_y.reshape(-1, 1, len(name_to_int))
    test_y = test_y.reshape(-1, 1, len(name_to_int))

    # create model
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = prepare_lstm_model(input_shape, len(name_to_int), _n_outputs)

    print(train_X.shape, train_y.shape)
    acc_list, lost_list = [], []
    start_time = datetime.now()
    train_history = model.fit(train_X, train_y, epochs=_epochs, verbose=1, batch_size=500)

    runtime_duration = (datetime.now() - start_time)
    runtime_duration_text = 'Training time: ' + str(runtime_duration).split('.')[0]
    print(f'Duration: {runtime_duration_text}')

    predict = model.predict(test_X)
    predict_result = EvaluateResult(predict, test_y, name_to_int)
    print(f' predict result {predict_result}')

    clear_predict = convert_back_y_predicted(predict, name_to_int)

    test_y = test_y.reshape(-1, len(name_to_int))

    # encode values of y
    test_y_decoded = []
    predict_decoded = []
    for i in range(0, len(test_y)):
        test_y_decoded.append(vector_to_label(test_y[i], int_to_name))
        predict_decoded.append(vector_to_label(clear_predict[i], int_to_name))

    test_y_decoded = np.asarray(test_y_decoded)
    predict_decoded = np.asarray(predict_decoded)

    real_test_values = [1 if test_y_decoded[i] == 'OK' else 0 for i in range(len(test_y_decoded))]
    predict_values = [1 if predict_decoded[i] == 'OK' else 0 for i in range(len(predict_decoded))]

    # Calculate the prediction scores
    acc = accuracy_score(real_test_values, predict_values)
    acc_text = ("Accuracy: %0.4f" % acc)
    p = precision_score(real_test_values, predict_values)
    precision_text = ("Precision: %0.4f" % p)
    r = recall_score(real_test_values, predict_values)
    recall_text = ("Recall: %0.4f" % r)
    f1 = f1_score(real_test_values, predict_values)
    f1_text = ("F1-score: %0.4f" % f1)


    '''Create plot chart for display results'''
    result_text = f'{acc_text}\n{precision_text}\n{recall_text}\n{f1_text}'
    plt.figure()
    plt.plot(train_history.history['accuracy'], label='Train accuracy')
    plt.plot(train_history.history['loss'], label='Train loss')
    plt.xlabel('Epochs', fontdict={'fontsize': 8})
    plt.title(f'Input steps: {_n_inputs}. Input features: {inputfeatures}. ',
              fontdict={'fontsize': 10, 'fontweight': 'medium'})
    plt.text(_epochs / 1.5, 0.65, result_text)
    plt.legend(loc='upper left', fontsize=8)
    file_name = f'{_n_inputs}'
    for feature in inputfeatures:
        file_name += f'-{feature}'

    result_values = (file_name, runtime_duration.seconds, acc, p, r, f1)
    print(result_values)

    file_name += '.png'
    file_name = GlobalVariables.evaluation_folder / file_name

    plt.savefig(file_name)
    cleanup_memory()
    return result_values


'''Run with Multivariates multisteps then evaluate the results'''
def run_evaluate():
    results = []
    start_program_time = datetime.now()
    for inputs in GlobalVariables.n_inputs:
        for cols in GlobalVariables.cols:
            try:
                result = train_predict(data, cols, inputs, 1, GlobalVariables.epochs)
                results.append(result)
            except Exception as e:
                print(f'Error: {e}')
            finally:
                pass

    columns = ['Model name', 'Running time', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df = pd.DataFrame(results, columns=columns)
    print(df)
    try:
        df.to_csv(GlobalVariables.evaluation_folder / 'results.csv', index=False)
    except Exception as e:
        print(f'Error: {e}')

    print(f'finished running programme in: {datetime.now()-start_program_time}')


if __name__=="__main__":
    run_evaluate()
