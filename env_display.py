"""
Visualize Small GridWorld using tkinter
Author: zhs
Date: Date: Sep 17, 2018
"""

import numpy as np
import tkinter as tk
import time

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class SmallGrid(tk.Tk, object):
    def __init__(self):
        super(SmallGrid, self).__init__()
        self.action_space = ['n', 'w', 's', 'e']
        self.title('small-gridworld')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.values = np.zeros((4, 4))  # store value function of each grid
        self.text_list = []
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # create 16 grids
        for col in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = [col, 0, col, MAZE_W * UNIT]
            self.canvas.create_line(x0, y0, x1, y1)

        for row in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = [0, row, MAZE_W * UNIT, row]
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # create terminal grid
        terminal1_center = origin
        self.terminal1 = self.canvas.create_rectangle(
            terminal1_center[0] - 20, terminal1_center[1] - 20,
            terminal1_center[0] + 20, terminal1_center[1] + 20,
            fill='yellow')

        terminal2_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.terminal2 = self.canvas.create_rectangle(
            terminal2_center[0] - 20, terminal2_center[1] - 20,
            terminal2_center[0] + 20, terminal2_center[1] + 20,
            fill='yellow')

        self.add_text()

        # pack all
        self.canvas.pack()

    def add_text(self):
        """display values on the board"""
        origin = np.array([20, 20])
        self.values = np.around(self.values, decimals=2)

        # update values of each grid
        for i in range(MAZE_W):
            for j in range(MAZE_H):
                textvar = str(self.values[i][j])
                text = self.canvas.create_text(origin[0] + i * UNIT, origin[1] + j * UNIT, text=textvar)
                self.text_list.append(text)

    def reset(self):
        """update values of the Small GridWorld"""
        self.update()
        time.sleep(0.5)

        for text in self.text_list:
            self.canvas.delete(text)

        self.add_text()


if __name__ == '__main__':
    s_g = SmallGrid()
    s_g.values = np.ones((4, 4))
    print(s_g.values)
    print(type(s_g.values))
    s_g.after(1000, s_g.reset())
    s_g.mainloop()
