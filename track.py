import pygame
import math
import numpy as np
from uosc.server import parse_message
import socket
from nnfs_model import *
import os.path
import time
import matplotlib.pyplot as plt
import multiprocessing
import sys

np.set_printoptions(precision=3, floatmode="fixed", suppress=True, linewidth=200)

class Track:
    def __init__(self, filename):
        self.cumulative_angle_average = np.array([])
        self.firsttrack = []
        self.width = 1000
        self.height = 1000
        self.margin = 100
        self.startingx = 0
        self.startingy = 0
        self.angle = 0
        self.scale_factor = 0
        self.filename = filename
        self.load_track(self.filename)
        self.offset_x = 0
        self.offset_y = 0

        self.sensor_inputs = 12
        self.output_neurons = 26
        self.layer_neurons = 60
        self.modelX = np.array([])
        self.modely = np.array([])
        self._init_model()

        self.sensor_queue = multiprocessing.Queue()
        self.full_round_queue = multiprocessing.Queue()
        self.model_parameters_queue = multiprocessing.Queue()
        self.rounds_traveled = 0

    def update_track(self, track_array):
        gyro_array = track_array[:, [10]]
        gyro_factor = 1
        
        cumulative_angle = np.cumsum(gyro_array * gyro_factor)
        print(gyro_factor,cumulative_angle[-1])
        while abs(cumulative_angle[-1] - 360) > 0.2 or abs(cumulative_angle[-1] < 0.2):
            if (cumulative_angle[-1] > 360): #or (cumulative_angle[-1] < 0):
                gyro_factor += 0.01
            else:
                gyro_factor -= 0.01
            cumulative_angle = np.cumsum(gyro_array * gyro_factor)
            #print(gyro_factor,cumulative_angle[-1])
        
        if self.cumulative_angle_average.size == 0:
            self.cumulative_angle_average = cumulative_angle
            self.firsttrack = self.cumulative_angle_average

        if len(cumulative_angle) < len(self.cumulative_angle_average):
            new_length = min(len(cumulative_angle), len(self.cumulative_angle_average))
            indices_new = np.linspace(0, len(self.cumulative_angle_average) - 1, new_length)
            self.cumulative_angle_average = np.interp(indices_new, np.arange(len(self.cumulative_angle_average)), self.cumulative_angle_average)
            
        elif len(cumulative_angle) > len(self.cumulative_angle_average):
            new_length = min(len(cumulative_angle), len(self.cumulative_angle_average))
            indices_new = np.linspace(0, len(cumulative_angle) - 1, new_length)
            cumulative_angle = np.interp(indices_new, np.arange(len(cumulative_angle)), cumulative_angle)

        similarity = self._check_track_similarity(cumulative_angle, self.firsttrack)
        #print(similarity)
        if similarity > 0.97:
            self.cumulative_angle_average = (cumulative_angle + self.cumulative_angle_average) / 2
        
        self._update_scale_and_offset_factor()
        return self.cumulative_angle_average

    def draw_track(self, screen):
        x = self.startingx
        y = self.startingy
        for i in range(len(self.cumulative_angle_average)):
            radians = np.radians(self.cumulative_angle_average[i])
            
            delta_x = self.scale_factor * np.cos(radians)
            delta_y = self.scale_factor * np.sin(radians)
            xprev = x
            yprev = y
            x -= delta_x
            y += delta_y
            
            offset_startpoint = (x + self.offset_x, y + self.offset_y)
            offset_endpoint = (xprev + self.offset_x, yprev + self.offset_y)

            pygame.draw.line(screen, (255, 255, 255), offset_startpoint, offset_endpoint)

    def draw_train(self, screen, location):
        x = self.startingx
        y = self.startingy

        loopyloop = int(location / self.output_neurons * len(self.cumulative_angle_average))

        for i in range(loopyloop):
            radians = np.radians(self.cumulative_angle_average[i])
            
            delta_x = self.scale_factor * np.cos(radians)
            delta_y = self.scale_factor * np.sin(radians)

            x -= delta_x
            y += delta_y
        offset_trainpoint = (x + self.offset_x, y + self.offset_y)
            
        pygame.draw.circle(screen, (255, 0, 255), offset_trainpoint, 10, 0)

    def save_track(self):
        np.save(self.filename, self.cumulative_angle_average)

    def load_track(self, filename):
        if os.path.isfile(filename):
            self.cumulative_angle_average = np.load(filename, allow_pickle=True)

    def update_train_model(self, new_round_array):
        new_round_array = np.array(new_round_array)
        np.linspace(0, 1, self.output_neurons)
        steps = np.linspace(0, 1, self.output_neurons)

        count = 0
        originalshape = new_round_array.shape
        scaled_steps = new_round_array.shape[0] * steps
        for step in scaled_steps[:-1]:
            self.modelX = np.append(self.modelX, new_round_array[int(step)])
            self.modely = np.append(self.modely, count)
            count += 1
        self.modely = np.array(self.modely).astype('uint8')
        self.modelX = self.modelX.reshape((len(self.modelX)//originalshape[1]),originalshape[1])    
        self.model.train(self.modelX, self.modely, epochs=450, batch_size=None, print_every=10000) #validation_data=(self.modelX, self.modely),
        self.model_parameters_queue.put(self.model.get_parameters())

    def receive_sensor_data(self, sock):
        value = None
        distance_right_norm = 0
        distance_left_norm = 0
        start_recording_learningdata = 0
        new_round_threshold_reached = 0
        gate_last_measured = time.time()
        
        track_data_list = []
        data_array = []
        
        while True:
            try:
                pkt = sock.recv(200)
                if not pkt:
                    addr, value = None, None
                else:
                    addr, tags, value = parse_message(pkt)
            except:
                addr, value = None, None

            if addr == "/sensors":
                data_array = value
                distance_right_norm = data_array[0]
                distance_left_norm = data_array[1]
                magnetox_norm = data_array[2]
                magnetoy_norm = data_array[3]
                magnetoz_norm = data_array[4]
                accelx_norm = data_array[5]
                accely_norm = data_array[6]
                accelz_norm = data_array[7]
                gyrox_norm = data_array[8]
                gyroy_norm = data_array[9]
                gyroz_norm = data_array[10]
                speed_norm = data_array[11]

                self.sensor_queue.put(data_array)

                if start_recording_learningdata == 1:
                    track_data_list.append(data_array)
                
                if distance_left_norm < -.96 and distance_right_norm < -0.96:
                    if time.time() - gate_last_measured > 3:
                        print("pillars")
                        new_round_threshold_reached = 1
                        gate_last_measured = time.time()
                    start_recording_learningdata = 1

                elif new_round_threshold_reached == 1:
                    print("new round")
                    new_round_threshold_reached = 0
                    
                    if self.rounds_traveled > 0:
                        self.full_round_queue.put(track_data_list)

                    track_data_list = []   
                    self.rounds_traveled += 1

    def _init_model(self):
        self.model = Model()
        self.model.add(Layer_Dense(self.sensor_inputs, self.layer_neurons))
        self.model.add(Activation_ReLU())
        self.model.add(Layer_Dense(self.layer_neurons, self.layer_neurons))
        self.model.add(Activation_ReLU())
        self.model.add(Layer_Dense(self.layer_neurons, self.output_neurons))
        self.model.add(Activation_Softmax())
        self.model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=5e-4), accuracy=Accuracy_Categorical())
        self.model.finalize()

    def _check_track_similarity(self, arr1, arr2):
        if len(arr1) < len(arr2):
            new_length = min(len(arr1), len(arr2))
            indices_new = np.linspace(0, len(arr2) - 1, new_length)
            arr2 = np.interp(indices_new, np.arange(len(arr2)), arr2)

        elif len(arr1) > len(arr2):
            new_length = min(len(arr1), len(arr2))
            indices_new = np.linspace(0, len(arr1) - 1, new_length)
            arr1 = np.interp(indices_new, np.arange(len(arr1)), arr1)

        #cos_sim = np.dot(arr2, arr1) / (np.linalg.norm(arr2) * np.linalg.norm(arr1)) #another way to compare similarity
        corr_coef = np.corrcoef(arr1, arr2)[0, 1]
        return(corr_coef)

    def _update_scale_and_offset_factor(self):
        x = self.startingx
        y = self.startingy
        initial_scale_factor = 1
        x_list = []
        y_list = []

        for i in range(len(self.cumulative_angle_average)):
            radians = np.radians(self.cumulative_angle_average[i])
            delta_x = initial_scale_factor * np.cos(radians)
            delta_y = initial_scale_factor * np.sin(radians)
            x -= delta_x
            y += delta_y
            x_list = np.append(x_list, x)
            y_list = np.append(y_list, y)

        x_min, x_max = np.min(x_list), np.max(x_list)
        y_min, y_max = np.min(y_list), np.max(y_list)
        track_width = x_max - x_min
        track_height = y_max - y_min
        x_scale = (self.width - self.margin * 2) / track_width
        y_scale = (self.height - self.margin * 2) / track_height
        scale = min(x_scale, y_scale)

        x_offset = (self.width - track_width * scale) / 2 - x_min * scale - self.margin / 2
        y_offset = (self.height - track_height * scale) / 2 + y_min * scale + self.margin / 2

        self.scale_factor = scale
        self.offset_x = x_offset 
        self.offset_y = y_offset 
        return

def main():
    multiprocessing.set_start_method("fork")
    track = Track("data/track.npz")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_ip = socket.getaddrinfo("0.0.0.0", 9000, socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_ip[0][4])
    sock.setblocking(0)
    
    pygame.init()
    screen = pygame.display.set_mode((track.width, track.height))
    clock = pygame.time.Clock()
    FPS = 60

    location = 0
    sensor_data = []
    full_round_list= []

    get_sensor_data_process = multiprocessing.Process(target=track.receive_sensor_data,args=(sock,))
    get_sensor_data_process.start()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not track.model_parameters_queue.empty():
            track.model.set_parameters(track.model_parameters_queue.get())
        if not track.sensor_queue.empty():
            sensor_data = np.array(track.sensor_queue.get())
            confidences = track.model.predict(sensor_data)
            predictions = track.model.output_layer_activation.predictions(confidences)

            if(np.max(confidences) > 0.9):
                location = predictions[0]
            #print(confidences)
            
        if not track.full_round_queue.empty():
            full_round_list = np.array(track.full_round_queue.get())
            track.update_track(full_round_list)
            NN_training_process = multiprocessing.Process(target=track.update_train_model,args=(full_round_list,))
            if not NN_training_process.is_alive():
                NN_training_process.start()
        
        screen.fill((0,0,0))       
        track.draw_train(screen, location)
        track.draw_track(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit

if __name__ == "__main__":  
    main()

"""
TODO: check track similarity also before re-training NN



"""