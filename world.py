import numpy as np
import cv2
import random
import heapq
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt


directions = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class World:
    def __init__(self, h = 900, w = 1600, water = 0.7):
        self.height = h;
        self.width = w;
        self.water = water
        self.world = np.array([[random.randint(1, 255) for _ in range(self.width)] for _ in range(self.height)], np.uint8);
 
    def create_path(self, k, h, w):
        heap = list()
        heapq.heappush(heap, (int(self.world[h][w]), h, w))
        
        while(len(heap) > 0 and k > 0):
                v, y, x = heapq.heappop(heap)
                k -= 1
                self.world[y][x] = 0
                for dx, dy in directions:
                    cx, cy = x + dx, y + dy
                    if ((cx <= 50 or cx >= self.width-50)
                        or (cy <= 50 or cy >= self.height-50)
                        or self.world[cy][cx] == 0): 
                        continue;
                    heapq.heappush(heap, (int(self.world[cy][cx]), cy, cx))
        return k
        
    def let_there_be_light(self, n_focus = 5):
        k = int(self.height*self.width*((1 - self.water)/(2*n_focus)))
        min_w, min_h = int(self.width * .15), int(self.height * .15)
        
        for _ in tqdm(range(n_focus)):
            w, h = random.randint(min_w, self.width-1 - min_w), random.randint(min_h, self.height-1-min_h)
            self.create_path(k, h, w)
        
        for i in range(self.height):
            for j in range(self.width):
                if (self.world[i][j] == 0):
                    self.world[i][j] = 1
                else:
                    self.world[i][j] = 0
                    
        self.world = cv2.dilate(self.world, (7, 7))

        cnt, h = cv2.findContours(self.world, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(cnt)):
            cv2.fillPoly(self.world, pts = [cnt[i]], color = 255)
        
        self.world = cv2.erode(self.world, (7, 7))

        for i in range(1, self.height - 1):
            for j in range(1, self.width -1):
                if (self.world[i][j] == 255):
                    self.world[i][j] = random.randint(1, 50)
                    
        self.world = cv2.GaussianBlur(self.world, (3, 3), 1, 1)
        self.world[0][0] = 255
        
        
    def show_world(self):
        #cv2.drawContours(self.world, cnt, -1, 255, 3);
        cv2.imshow("za warudo", self.world);
        cv2.waitKey(0);
        cv2.destroyAllWindows()
        
    def save(self):
        cv2.imwrite("map.png", self.world)
        
    def plot3d(self):
        z = self.world
        x = np.linspace(0, self.width-1, self.width)
        y = np.linspace(0, self.height-1, self.height)
        x, y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                            linewidth=0, antialiased=False, shade=False)
        plt.show()
        
if __name__ == "__main__":
    world = World(500, 500)
    world.let_there_be_light()
    world.save()
    world.plot3d()