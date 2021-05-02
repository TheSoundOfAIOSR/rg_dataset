import sys
import threading
from io import StringIO
from multiprocessing.pool import ThreadPool
import concurrent.futures
import queue
import lorem

from active_learning.pool_based_AL import ActiveLearning
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QPlainTextEdit,
    QApplication,
    QHBoxLayout,
    QFormLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
    QLabel,
)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Learning Annotator")


        # create an outer layout
        self.outerLayout = QVBoxLayout()

        # create a form layout for the n_instances and N_QUERIES
        self.topLayout = QHBoxLayout()
        # add labels and line edits to the form layout
        self.inst_label= QLabel('nb of instances')
        self.query_label= QLabel('nb of queries')
        self.inst_edit = QLineEdit()
        self.query_edit = QLineEdit()

        self.topLayout.addWidget(self.inst_label)
        self.topLayout.addWidget(self.inst_edit)
        self.topLayout.addWidget(self.query_label)
        self.topLayout.addWidget(self.query_edit)

        # create a button layout for first training phase
        self.start_layout = QVBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.run_alScript)
        self.start_layout.addWidget(self.start_button)


        # create a layout for training progress
        self.console_layout = QVBoxLayout()
        # create a label for duration of training
        self.console_pt = QPlainTextEdit()
        self.console_layout.addWidget(self.console_pt)



        # nest the inner layout into the outer layout
        self.outerLayout.addLayout(self.topLayout)
        self.outerLayout.addLayout(self.start_layout)
        self.outerLayout.addLayout(self.console_layout)
        self.outerLayout.setContentsMargins(5, 0, 0, 5)

        self.setLayout(self.outerLayout)


    # def run_alScript(self):
    #     nb_instances = self.inst_edit.text()
    #     nb_queries = self.query_edit.text()
    #     al = ActiveLearning(n_queries=nb_queries, n_instances=nb_instances)
    #
    #     for _, sentences in al.query_sentences():
    #         print(sentences)


class Worker(QObject):
    finished = pyqtSignal()
    def __init__(self, nb_queries, nb_instances):
        self.nb_queries = nb_queries
        self.nb_instances = nb_instances

    def run_alScript(self):
        nb_instances = self.inst_edit.text()
        nb_queries = self.query_edit.text()
        al = ActiveLearning(n_queries=self.nb_queries, n_instances=self.nb_instances)

        for _, sentences in al.query_sentences():
            print(sentences)
        self.finished.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())