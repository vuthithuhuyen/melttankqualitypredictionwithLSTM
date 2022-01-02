import sys
from pathlib import Path

from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QMainWindow, QFileSystemModel, QSlider, QLabel
from PyQt5.uic import loadUi

from Controller.MainWindowController import Selected_Row2DF, TabelViewClick, TrainClick, DisplayPlot, PredictRegidClick, \
    LoadModelClick, PredictFlexibleClick, AutoGenerateParameters, DisplayHist
from Helper.ControllerHelper import sliderChangeValue, InitialComboxState
from Helper.DataPreprocessing import ReadData
from Helper.MyPathFunctions import GetCWD
from Helper.SystemHelper import ExitProgram
from Helper.Table import CsvTableModel, DfTableViewModel
from Model import GlobalVariables
from Model.GlobalVariables import mainWindowUI, raw_data_file


class MainWindowClass(QMainWindow):
    def __init__(self):
        try:
            super(MainWindowClass, self).__init__()
            loadUi(mainWindowUI, self)

            # # Slider split train/test value
            self.sld_SplitTrainTest.valueChanged.connect(
                lambda: sliderChangeValue(self.sld_SplitTrainTest, self.lblSplitValue, True))

            self.sld_MixA_Temp.valueChanged.connect(
                lambda: self.sliderFlexibleChangeValue('A'))
            self.sld_MixB_Temp.valueChanged.connect(
                lambda: self.sliderFlexibleChangeValue('B'))




            # Load CSV file to table
            df, GlobalVariables.X, GlobalVariables.y = ReadData()
            self.table_model = DfTableViewModel(df)

            self.tableView.setSortingEnabled(False)
            self.tableView.setModel(self.table_model)


            # Display plot data:
            DisplayHist(df, 1, self.lbl_image1, '#12B28C')
            DisplayHist(df, 2, self.lbl_image2, '#C4860B')
            DisplayHist(df, 3, self.lbl_image3, '#9D1F45')
            DisplayHist(df, 4, self.lbl_image4, '#008EFF')

            # Init combobox
            numberInputs = [str(x) for x in range(1, 10)]
            InitialComboxState(self.cmbNumberInputs, numberInputs)


            # Click row on table:
            self.tableView.clicked.connect(lambda: TabelViewClick(self, self.tableView, self.tblRegidPredict))

            # # Button train
            self.btnTrain.clicked.connect(lambda: TrainClick(self, self.lblSplitValue, self.lbl_image3, self.txtEpoch))

            # button load trained model:
            self.btnLoadmodel.clicked.connect(lambda: LoadModelClick(self))

            # #Button predict
            self.btnRegidPredict.clicked.connect(lambda: PredictRegidClick(self.model, self.lbl_Regrid_Predict_Val))
            #
            # # Button Flexible Click:
            # self.btnFlixiblePredict.clicked.connect(lambda: PredictFlexibleClick(self.cmbMixA_State, self.cmbMixB_State,
            #                                                                      self.sld_MixA_Temp, self.sld_MixB_Temp,
            #                                                                      self.model, self.lbl_Flexible_Predict_Val))

            # Button autogenerate prediction:
            self.btnAutoGenerate.clicked.connect(lambda: AutoGenerateParameters(self.model))


        except Exception as e:
            print(e)

    # slider flexible change value
    def sliderFlexibleChangeValue(self, sliderName):
        try:
            if sliderName == 'A':
                val = self.sld_MixA_Temp.value()
                self.lbl_MixA_Temp_Val.setText(str(val))
            else:
                val = self.sld_MixB_Temp.value()
                self.lbl_MixB_Temp_Val.setText(str(val))

            if GlobalVariables.load_trained:
                PredictFlexibleClick(self.cmbMixA_State, self.cmbMixB_State,
                                     self.sld_MixA_Temp, self.sld_MixB_Temp,
                                     self.model, self.lbl_Flexible_Predict_Val)
        except Exception as e:
            print(e)

