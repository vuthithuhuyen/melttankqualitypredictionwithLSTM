import cv2
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from PyQt5.QtWidgets import QLabel, QTableView, QPlainTextEdit, QComboBox, QSlider
from sklearn.model_selection import train_test_split

from Helper.DataPreprocessing import ScaleData
from Helper.MathHelper import RoundPredictValue
from Helper.OpenCVHelper import rescale_frame
from Helper.Plot import PandasColToPlot, PandasColToHist
from Helper.PyQTHelper import QLabelDisplayImage
from Helper.TryExept import Information, ErrorMessage
from Helper.pandasTable import PandasTableModel
from Model import GlobalVariables
from Model.DeeplearningModel import prepare_mlp_model, GetReverseYValue, SaveDictToFile, LoadFileToDict
from Model.GlobalVariables import scaler


def Selected_Row2DF(tbl: QTableView):
    try:
        indexes = tbl.selectionModel().selectedIndexes()
        listRows = []

        for index in sorted(indexes):
            listRows.append(index.row())
        index = listRows[0]

        # Get row data
        row = GlobalVariables.analysis_data.iloc[index, 1: -1]
        row_df = pd.DataFrame(row)
        row_df = row_df.T
        GlobalVariables.row_to_predict = row_df.copy()

        return row_df

    except Exception as e:
        print(e)
        return None


def TabelViewClick(self, tblView: QTableView, tblRegdid: QTableView):
    try:
        df = Selected_Row2DF(tblView)
        self.regid_model = PandasTableModel(df)
        tblRegdid.setModel(self.regid_model)

        if GlobalVariables.load_trained:
            PredictRegidClick(self.model, self.lbl_Regrid_Predict_Val)
    except Exception as e:
        print(e)


# load trained model:
def LoadModelClick(self):
    try:
        self.model = tensorflow.keras.models.load_model('TRAINED_MODEL')
        GlobalVariables.scaler = joblib.load('SCALER.gz')
        GlobalVariables.int_to_name = LoadFileToDict('INT_TO_NAME')
        GlobalVariables.name_to_int = LoadFileToDict('NAME_TO_INT')

        GlobalVariables.load_trained = True
        Information(self, 'Trained model and parameters were loaded successfully!')
    except Exception as e:
        ErrorMessage(str(e))


# Save trained model:
def SaveTrainModel(_model, _name_to_int, _int_to_name, _scaler):
    _model.save('TRAINED_MODEL')
    # Save the scaler:
    joblib.dump(GlobalVariables.scaler, 'SCALER.gz')
    SaveDictToFile(_int_to_name, 'INT_TO_NAME')
    SaveDictToFile(_name_to_int, 'NAME_TO_INT')


# Train model
def TrainClick(self, split: QLabel, lblDisplay: QLabel, txtEpoch: QPlainTextEdit):
    try:
        epoch = txtEpoch.toPlainText()
        epoch = int(epoch)

        split_val = float(split.text())
        test_size = round(1 - split_val, 2)
        print(f'test size: {test_size}')
        X, y, GlobalVariables.name_to_int, GlobalVariables.int_to_name = ScaleData(GlobalVariables.X, GlobalVariables.y)

        # split train, test
        split = train_test_split(X, y, test_size=test_size)
        (train_X, test_X, train_y, test_y) = split

        # create model
        n_features = 3
        self.model = prepare_mlp_model(n_features, GlobalVariables.name_to_int)
        print(self.model.summary())
        print(train_X.shape, train_y.shape)

        # Plot history -> Display
        history = self.model.fit(train_X, train_y, epochs=epoch, verbose=2, batch_size=30)
        DisplayTrainingHistory(history, lblDisplay)

        # Save model to files:
        SaveTrainModel(self.model, GlobalVariables.name_to_int, GlobalVariables.int_to_name, scaler)

        Information(self, 'Model was trained successfully!')

    except Exception as e:
        print(e)
        ErrorMessage(self, str(e))
        return


# Plot history and disply on QLabel
def DisplayTrainingHistory(history, lblDisplay: QLabel):
    try:
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.legend()
        filename = 'training_history.png'
        plt.savefig(filename)
        img = cv2.imread(filename)
        img = rescale_frame(img, 50)
        QLabelDisplayImage(lblDisplay, img)
    except Exception as e:
        print(e)


# Display plot data of data frame:
def DisplayPlot(df: pd.DataFrame, col: int, label: QLabel, color):
    plt = PandasColToPlot(df, col, color)
    filename = f'plot_{str(col)}.png'
    plt.savefig(filename)

    img = cv2.imread(filename)
    img = rescale_frame(img, 50)
    QLabelDisplayImage(label, img)


# Display hist Plot:
def DisplayHist(df: pd.DataFrame, col: int, label, color):
    plt = PandasColToHist(df, col, color)
    filename = f'hist_{str(col)}.png'
    plt.savefig(filename)

    img = cv2.imread(filename)
    img = rescale_frame(img, 50)
    QLabelDisplayImage(label, img)


# Prediction. X => y
def Prediction(model, X):
    try:
        X = X.values
        X_transformed = GlobalVariables.scaler.transform(X)
        y = model.predict(X_transformed)
        predict = GetReverseYValue(y, GlobalVariables.int_to_name)
        return X, y, predict
    except Exception as e:
        print(e)


# Predict with Regrid
def PredictRegidClick(model, label: QLabel):
    try:
        if (GlobalVariables.row_to_predict is None) or (model is None):
            print('Nothing to predict!')
            return

        X = GlobalVariables.row_to_predict.copy()
        X, y, predict = Prediction(model, X)
        y = RoundPredictValue(y[0])

        output = f'Input: {X[0]} ==> {y} ==> {predict}'
        label.setText(output)
    except Exception as e:
        print(e)


# Predict with flexible input
def PredictFlexibleClick(cmb_mixA: QComboBox, cmb_mixB: QComboBox, slider_mixA: QSlider, slider_mixB: QSlider,
                         model, label: QLabel):
    try:
        if model is None:
            print('Nothing to predict!')
            return

        if not GlobalVariables.load_trained:
            print('Please load trained model first!')
            return

        X = GetFlexibleParameters(cmb_mixA, cmb_mixB, slider_mixA, slider_mixB)
        X, y, predict = Prediction(model, X)
        y = RoundPredictValue(y[0])

        output = f'Input: {X[0]} ==> {y} ==> {predict}'
        label.setText(output)

    except Exception as e:
        print(e)


# Get parameters from comboboxes and sliders:
def GetFlexibleParameters(cmb_mixA: QComboBox, cmb_mixB: QComboBox, slider_mixA: QSlider, slider_mixB: QSlider):
    try:
        mixA_state = cmb_mixA.currentText()
        mixB_state = cmb_mixB.currentText()
        mixA_temprature = slider_mixA.value()
        mixB_temprature = slider_mixB.value()
        row = [int(mixA_state), int(mixB_state), mixA_temprature, mixB_temprature]
        df = pd.DataFrame(row)
        df = df.T
        return df
    except Exception as e:
        print(e)


# Auto generate parameters:
def AutoGenerateParameters(model):
    try:
        mixA_state = [0]
        mixB_state = [0, 1]
        mixA_temprature = [60]
        mixB_temprature = [i for i in range(30, 80)]

        predict_file = f'predict_results.csv'
        result = []
        for para1 in mixA_state:
            for para2 in mixB_state:

                for para3 in mixA_temprature:
                    for para4 in mixB_temprature:
                        row = [para1, para2, para3, para4]
                        df = pd.DataFrame(row)
                        X = df.T
                        X, y, predict = Prediction(model, X)
                        row.append(predict)
                        result.append(row)

        df = pd.DataFrame(result)
        df.to_csv(predict_file)
        print('Done')
    except Exception as e:
        print(e)