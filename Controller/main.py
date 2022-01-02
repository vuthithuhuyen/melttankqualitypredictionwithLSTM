import sys

from PyQt5.QtWidgets import QApplication

from Model.MainWindowModel import MainWindowClass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindowClass()
    window.show()
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(e)