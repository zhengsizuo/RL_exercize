"""
Environment of flappy birds using pygame module.
Author: zhs
Date: Oct 30, 2018
"""

import pygame
import time
import numpy as np

# 设置屏幕为10*10=100个状态
width = 10
height = 10
# 两列障碍物的x坐标
first_col = 120
second_col = 280
# 绘制四列障碍物所需的参数
gap_distance = 90
col_one = 3
col_two = 4
col_three = 5
col_four = 2


class Bird(object):
    def __init__(self, screen):
        self.screen = screen
        self.image = pygame.image.load('images/bird.png')  # 40*30
        self.rect = self.image.get_rect()
        self.rect.topleft = np.array([0, 0])

    def state_to_coordinate(self, state):
        """状态到坐标的转化函数，便于绘制小鸟"""
        x = (state % width) * 40
        y = (state // width) * 30
        return np.array([x, y])

    def blitme(self):
        """渲染图片函数"""
        self.screen.blit(self.image, self.rect)


class BgSet(object):
    def __init__(self, screen):
        self.screen = screen
        # 加载背景及障碍物图像，并设置其rect属性
        self.bg_image = pygame.image.load('images/background.png')  # 400*400
        self.obstacle_image = pygame.image.load('images/obstacle.png')  # 40*30
        self.ob_rect = self.obstacle_image.get_rect()

    def draw_obstacles(self, ob_pos, col):
        """根据左上角坐标连续绘制障碍物"""
        for i in range(1, col):
            ob_pos[1] += self.ob_rect.height
            self.screen.blit(self.obstacle_image, ob_pos)

    def blitme(self):
        """绘制整个游戏背景"""
        bg_pos = np.array([0, 0])
        ob_pos = np.array([first_col, 0])
        self.screen.blit(self.bg_image, bg_pos)

        self.screen.blit(self.obstacle_image, ob_pos)
        self.draw_obstacles(ob_pos, col_one)
        ob_pos[1] += gap_distance
        self.draw_obstacles(ob_pos, col_two+1)

        ob_pos = np.array([second_col, 0])
        self.screen.blit(self.obstacle_image, ob_pos)
        self.draw_obstacles(ob_pos, col_three)
        ob_pos[1] += gap_distance
        self.draw_obstacles(ob_pos, col_four+1)


def run_env():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Find you")
    bird_male = Bird(screen)
    bird_female = Bird(screen)
    bird_female.rect.topleft = np.array([360, 0])
    bg_set = BgSet(screen)

    while True:
        bg_set.blitme()
        bird_male.blitme()
        bird_female.blitme()

        pygame.display.update()
        time.sleep(0.1)


if __name__ == '__main__':
    run_env()