from PyQt5.QtWidgets import QMessageBox


def Information(self, msg):
    QMessageBox.information(self, "Information!", msg)


def ErrorMessage(self, msg):
    QMessageBox.warning(self, "Warning!", msg)