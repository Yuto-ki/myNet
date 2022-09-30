import matplotlib.pyplot as plt
import math


col_n = 4


class OutPutter:
    def __init__(self, graph_n, title):
        row_n = math.ceil(graph_n / col_n)
        self.fig, self.ax = plt.subplots(nrows=row_n, ncols=col_n, constrained_layout=True, linewidth=1)
        self.fig.suptitle(title)
        self.ax = self.ax.ravel()

    def edit_graph(self, number, title, data_x, data_y1, data_y2):
        self.ax[number].set_title(title)
        self.ax[number].plot(data_x, data_y1)
        self.ax[number].plot(data_x, data_y2)

    def save_fig(self, file_name):
        self.fig.savefig(file_name)
