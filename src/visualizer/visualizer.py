import numpy as np
import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self):
        self.__xlim: tuple = (0, 1000)
        self.__ylim: tuple = (-1000, 1000)
        self.__plt = plt

    def prepare(self):
        t = np.zeros(100)
        y = np.zeros(100)
        self.__plt.ion()
        self.__fig, (self.__ax1, self.__ax2, self.__ax3) = \
            self.__plt.subplots(ncols=3, figsize=(15, 5))
        self.__ax1.set_xlim(*self.__xlim)
        self.__ax2.set_xlim(*self.__xlim)
        self.__ax3.set_xlim(*self.__xlim)
        self.__ax1.set_ylim(*self.__ylim)
        self.__ax2.set_ylim(*self.__ylim)
        self.__ax3.set_ylim(*self.__ylim)

    def update_chart(self, xs, accs, accs_global, time_:float):
        self.__ax1.cla()
        self.__ax2.cla()
        self.__ax3.cla()

        self.__ax1.plot(
            list(range(len(xs))),
            np.array(xs).T[0],
            linewidth=1.,
            label="roll"
        )
        self.__ax1.plot(
            list(range(len(xs))),
            np.array(xs).T[1],
            linewidth=1.,
            label="pitch"
        )
        # self.__ax1.plot(
        #     list(range(len(xs))),
        #     np.array(xs).T[2],
        #     linewidth=1.,
        #     label="yaw"
        # )
        self.__ax1.set_title('Pose of JoyCon: {:.2f} sec'.format(time_))
        self.__ax1.set_xlim(self.__xlim)
        self.__ax1.legend(
            bbox_to_anchor=(0, 1),
            loc='upper left',
            borderaxespad=0,
            fontsize=13
        )

        self.__ax2.plot(
            list(range(len(accs))),
            np.array(accs).T[0],
            linewidth=1.,
            label="x_local"
        )
        self.__ax2.plot(
            list(range(len(accs))),
            np.array(accs).T[1],
            linewidth=1.,
            label="y_local"
        )
        self.__ax2.plot(
            list(range(len(accs))),
            np.array(accs).T[2],
            linewidth=1.,
            label="z_local"
        )
        self.__ax2.set_title('Acc of JoyCon: {:.2f} sec'.format(time_))
        self.__ax2.set_xlim(self.__xlim)
        self.__ax2.legend(
            bbox_to_anchor=(0, 1),
            loc='upper left',
            borderaxespad=0,
            fontsize=13
        )

        self.__ax3.plot(
            list(range(len(accs_global))),
            np.array(accs_global).T[0],
            linewidth=1.,
            label="x_global"
        )
        self.__ax3.plot(
            list(range(len(accs_global))),
            np.array(accs_global).T[1],
            linewidth=1.,
            label="y_global"
        )
        self.__ax3.plot(
            list(range(len(accs_global))),
            np.array(accs_global).T[2],
            linewidth=1.,
            label="z_global"
        )
        self.__ax3.set_title('Acc of JoyCon(global): {:.2f} sec'.format(time_))
        self.__ax3.set_xlim(self.__xlim)
        self.__ax3.legend(
            bbox_to_anchor=(0, 1),
            loc='upper left',
            borderaxespad=0,
            fontsize=13
        )

        # グラフを表示する
        self.__plt.pause(1/50)
