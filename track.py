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
        self.track_xcoords = np.array([])
        self.track_ycoords = np.array([])

        self.firsttrack = []
        self.width = 1000
        self.height = 1000
        self.margin = 150
        self.startingx = 0
        self.startingy = 0
        self.angle = 0
        self.scale_factor = 0
        self.filename = filename
        self.load_track(self.filename)
        self.offset_x = 0
        self.offset_y = 0

        self.sensor_inputs = 12
        self.output_neurons = 30
        self.layer_neurons = 60
        self.modelX = np.array([])
        self.modely = np.array([])
        self._init_model()

        self.sensor_queue = multiprocessing.Queue()
        self.full_round_queue = multiprocessing.Queue()
        self.model_parameters_queue = multiprocessing.Queue()
        self.rounds_traveled = 0

        self.first_round_data = np.array([])
        self.first_round_data_stored = 0

    def update_track(self, track_array):
        gyro_array = track_array[:, [10]] # only check gyro on Z axis
        gyro_factor = 0.5
        
        cumulative_angle = np.cumsum(gyro_array * gyro_factor)
        gyro_factor = 180 / cumulative_angle[-1]
        cumulative_angle = np.cumsum(gyro_array * gyro_factor)
        
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

        self.cumulative_angle_average = (cumulative_angle + self.cumulative_angle_average) / 2

        self._update_scale_and_offset_factor()

        x_coords = np.cumsum(np.cos(np.radians(self.cumulative_angle_average)))
        y_coords = np.cumsum(np.sin(np.radians(self.cumulative_angle_average)))

        x_correction = (x_coords[-1] - x_coords[0]) / len(x_coords)
        y_correction = (y_coords[-1] - y_coords[0]) / len(y_coords)

        x_coords_corrected = x_coords - np.arange(len(x_coords)) * x_correction
        y_coords_corrected = y_coords - np.arange(len(y_coords)) * y_correction

        x_coords_corrected = -x_coords_corrected * self.scale_factor # changed to - to mirror track
        y_coords_corrected = y_coords_corrected * self.scale_factor
        x_coords_corrected += self.offset_x
        y_coords_corrected += self.offset_y / 2

        self.track_xcoords = x_coords_corrected
        self.track_ycoords = y_coords_corrected

    def draw_track(self, screen):
        for i in range(len(self.track_xcoords) - 1):
            startpoint = (self.track_xcoords[i], self.track_ycoords[i])
            endpoint = (self.track_xcoords[i + 1], self.track_ycoords[i + 1])
            pygame.draw.line(screen, (255, 255, 255), startpoint, endpoint, 2)
                
    def draw_train(self, screen, location):
        loopyloop = int(location / self.output_neurons * len(self.track_xcoords))
        trainpoint = (self.track_xcoords[0], self.track_ycoords[0])
        for i in range(loopyloop):
            trainpoint = (self.track_xcoords[i], self.track_ycoords[i])
        pygame.draw.circle(screen, (255, 0, 255), trainpoint, 10, 0)

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

    def check_track_similarity(self, arr1, arr2):
        if len(arr1) < len(arr2):
            new_length = min(len(arr1), len(arr2))
            indices_new = np.linspace(0, len(arr2) - 1, new_length)
            arr2 = np.interp(indices_new, np.arange(len(arr2)), arr2)

        elif len(arr1) > len(arr2):
            new_length = min(len(arr1), len(arr2))
            indices_new = np.linspace(0, len(arr1) - 1, new_length)
            arr1 = np.interp(indices_new, np.arange(len(arr1)), arr1)

        cos_sim = np.dot(arr2, arr1) / (np.linalg.norm(arr2) * np.linalg.norm(arr1)) #another way to compare similarity
        corr_coef = np.corrcoef(arr1, arr2)[0, 1]
        return(cos_sim)

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

    def _update_scale_and_offset_factor(self):
        x_coords = np.cumsum(np.cos(np.radians(self.cumulative_angle_average)))
        y_coords = np.cumsum(np.sin(np.radians(self.cumulative_angle_average)))

        x_correction = (x_coords[-1] - x_coords[0]) / len(x_coords)
        y_correction = (y_coords[-1] - y_coords[0]) / len(y_coords)

        x_coords_corrected = x_coords - np.arange(len(x_coords)) * x_correction
        y_coords_corrected = y_coords - np.arange(len(y_coords)) * y_correction

        x_coords_corrected = -x_coords_corrected
        y_coords_corrected = y_coords_corrected

        x_min, x_max = np.min(x_coords_corrected), np.max(x_coords_corrected)
        y_min, y_max = np.min(y_coords_corrected), np.max(y_coords_corrected)
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
    font_large = pygame.font.SysFont('Arial', 30)
    font_small = pygame.font.SysFont('Arial', 15)
    text_surface_waiting = font_large.render('waiting for train to finish first complete round', True, (255, 255, 255))
    text_rect_waiting = text_surface_waiting.get_rect()
    text_rect_waiting.center = (track.width//2, track.height//2 - 100)

    location = 0
    sensor_data = []
    full_round_list= []

    get_sensor_data_process = multiprocessing.Process(target=track.receive_sensor_data,args=(sock,))
    get_sensor_data_process.start()

    similarity = 1 #for the first round I consider that the data is OK to use

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

            if(np.max(confidences) > 0.97):
                location = predictions[0]
             
        if not track.full_round_queue.empty():
            full_round_list = np.array(track.full_round_queue.get())

            if track.first_round_data_stored == 0:
                track.first_round_data = full_round_list
                track.first_round_data_stored = 1

            if track.first_round_data.shape[0] != 0:
                gyro_array_first_lap = track.first_round_data[:, [10]].flatten()
                gyro_array_current_lap = full_round_list[:, [10]].flatten()
                similarity = track.check_track_similarity(gyro_array_first_lap, gyro_array_current_lap)

            if similarity > 0.95:
                track.update_track(full_round_list)
                NN_training_process = multiprocessing.Process(target=track.update_train_model,args=(full_round_list,))
                if not NN_training_process.is_alive():
                    NN_training_process.start()
        
        screen.fill((0,0,0))       

        if track.first_round_data_stored == 1:
            track.draw_train(screen, location)
            track.draw_track(screen)

        else:
            screen.blit(text_surface_waiting, text_rect_waiting)

        text_surface_data = font_small.render(str(sensor_data), True, (255, 255, 255))
        text_rect_data = text_surface_waiting.get_rect()
        text_rect_data.center = (track.width //2, track.height - 50)
        screen.blit(text_surface_data, text_rect_data)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit
    get_sensor_data_process.join()

if __name__ == "__main__":  
    main()