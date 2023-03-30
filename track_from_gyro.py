
import numpy as np
import os
import pygame

np.set_printoptions(precision=3, floatmode="fixed", suppress=True, linewidth=400)


datapath = "data/new"
datafilelist = os.listdir(datapath)
#datafilelist = ["roundv2-2.npy","roundv2-3.npy"]

avg_arr=np.array([])
cumulative_angle_average = np.array([])
cumulative_angle = np.array([])

for file in datafilelist:
    filewithpath = datapath+"/"+file
    arr = np.load(filewithpath, allow_pickle=True)
    arr = arr[:, [10]]
    #window_size = 10
    #weights = np.repeat(1.0, window_size) / window_size
    #arr = np.apply_along_axis(lambda m: np.convolve(m, weights, 'valid'), axis=1, arr=arr)

    gyro_factor = 1
    cumulative_angle = np.cumsum(arr * gyro_factor)
    while abs(cumulative_angle[-1] - 360) > 0.2:
        if cumulative_angle[-1] > 360:
            gyro_factor += 0.01
        else:
            gyro_factor -= 0.01
        cumulative_angle = np.cumsum(arr * gyro_factor)

    
    if cumulative_angle_average.size == 0:
        cumulative_angle_average = cumulative_angle

    
    if len(cumulative_angle) < len(cumulative_angle_average):
        new_length = min(len(cumulative_angle), len(cumulative_angle_average))
        indices_new = np.linspace(0, len(cumulative_angle_average) - 1, new_length)
        cumulative_angle_average = np.interp(indices_new, np.arange(len(cumulative_angle_average)), cumulative_angle_average)
        
    elif len(cumulative_angle) > len(cumulative_angle_average):
        new_length = min(len(cumulative_angle), len(cumulative_angle_average))
        indices_new = np.linspace(0, len(cumulative_angle) - 1, new_length)
        cumulative_angle = np.interp(indices_new, np.arange(len(cumulative_angle)), cumulative_angle)

    cumulative_angle_average = (cumulative_angle + cumulative_angle_average) / 2


print(cumulative_angle_average)

width = 800
height = 600
margin = 10
x = 400
y = 100
startingx = x
startingy = y

angle = 0
def find_scale_factor(x, y, witdh, height, margin):
    scale_factor = 0
    ok = False
    while not ok:
        for i in range(len(cumulative_angle_average)):
            radians = np.radians(cumulative_angle_average[i])
            delta_x = scale_factor * np.cos(radians)
            delta_y = scale_factor * np.sin(radians)
            prevx = x
            prevy = y
            x -= delta_x
            y += delta_y
            maxx = max(x, prevx)
            minx = min(x, prevx)
            maxy = max(y, prevy)
            miny = min(y, prevy)
        
        widthmax = abs(maxx - (width - margin))
        widthmin = abs(minx - margin)
        heightmax = abs(maxy - (height - margin))
        heightmin = abs(miny - margin)
        print(widthmax)
        if widthmax < 5:
            ok = True
        else:
            scale_factor += 0.1
        
    return scale_factor

scale_factor = find_scale_factor(startingx,startingy,width, height,margin)

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Train Track")

for i in range(len(cumulative_angle_average)):
    radians = np.radians(cumulative_angle_average[i])
    
    delta_x = scale_factor * np.cos(radians)
    delta_y = scale_factor * np.sin(radians)
    xprev = x
    yprev = y
    x -= delta_x
    y += delta_y
    
    pygame.draw.line(screen, (255, 255, 255), (int(xprev), int(yprev)), (int(x), int(y)))
#pygame.draw.line(screen, (255, 255, 255), (int(startingx), int(startingy)), (int(x), int(y)))
pygame.display.flip()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()


distance_right_norm = 0
distance_left_norm = 1
magnetox_norm = 2
magnetoy_norm = 3
magnetoz_norm = 4
accelx_norm = 5
accely_norm = 6
accelz_norm = 7
gyrox_norm = 8
gyroy_norm = 9
gyroz_norm = 10
speed_norm = 11



