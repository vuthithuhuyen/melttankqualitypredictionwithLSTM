from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


from Helper.MyPathFunctions import GetCWD
import pathlib


appPath = GetCWD().parent
mainWindowUI = appPath / 'View' / 'main.ui'
mainwindow = appPath / 'View' / 'mainwindow.ui'


raw_data_file = appPath / 'Data' / 'melting_tank.csv'
evaluation_folder = appPath / 'Evaluation results'
full_cols = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'TAG'],


cols = [
    # ['MELT_TEMP'] + ['TAG'],
    # ['MOTORSPEED'] + ['TAG'],
    # ['MELT_WEIGHT'] + ['TAG'],
    # ['INSP'] + ['TAG'],

    ['MELT_TEMP', 'MOTORSPEED', 'INSP'] + ['TAG'],
    ['MELT_TEMP', 'MOTORSPEED'] + ['TAG'],
    ['MELT_TEMP', 'INSP'] + ['TAG'],
    ['MOTORSPEED', 'INSP'] + ['TAG'], # BEST!!!

]
test_size = 0.3
n_inputs = [1, 5, 10, 20]

epochs = 50

scaler = MinMaxScaler()
name_to_int, int_to_name = None, None
load_trained = False


analysis_data = None

X, y = None, None
display_rescale = 60
row_to_predict = None





