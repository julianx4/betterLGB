import pygame
import math
import numpy as np
from uosc.server import parse_message
import socket
from nnfs_model import *

class rail:
    def __init__(self, length = 0.0, radius = 0.0, angle = 0.0, direction = 0, ties = 0, color = None):
        self.length = length
        self.radius = radius
        self.angle = angle
        self.direction = direction
        self.ties = ties
        if length > 0:
            self.x = length
            self.y = 0
        else:
            self.x = abs(math.sin(math.radians(self.angle)) * self.radius)
            if self.direction < 0:
                self.y = self.radius - (math.cos(math.radians(self.angle)) * self.radius)
            if self.direction > 0:
                self.y = -(self.radius - (math.cos(math.radians(self.angle)) * self.radius))

    def show_rail(self):
        return self.length, self.radius, self.angle, self.direction

def read_packet(sock):
    try:
        pkt = sock.recv(200)
        if not pkt:
            return None, None
        else:
            addr, tags, value = parse_message(pkt)
            return addr, value
    except:
        return None, None

def draw_track(screen, starting_point, track, scale):
    angle = 180
    
    for i in range(0, len(track)):
        rot_angle = math.radians(angle + track[i].angle * track[i].direction)
        rot_matrix = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        angle = math.degrees(rot_angle)

        track_vector_unrot = np.array([track[i].x * scale, track[i].y * scale])
        track_vector = np.dot(rot_matrix, track_vector_unrot)
        end_point = np.add(starting_point, track_vector)
        if track[i].length == 0:
            pygame.draw.line(screen, (255, 255, 255), starting_point, end_point)
        else:
            pygame.draw.line(screen, (255, 255, 255), starting_point, end_point)
        starting_point = end_point

def train_position(starting_point, track, tie_count, scale): 
    angle = 180
    track_length = 0
    track_length_temp = 0
    elapsed_rail_ties = 0
    current_track = 0
    train_position = 0
    train_point = None

    for i in range(0, len(track)):
        track_length += track[i].ties
    train_position = tie_count % track_length

    for i in range(0, len(track)):
        track_length_temp += track[i].ties
        if track_length_temp > train_position:
            current_track = i
            break

    for i in range(0, len(track)):
        rot_angle = math.radians(angle + track[i].angle * track[i].direction)
        rot_matrix = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        angle = math.degrees(rot_angle)

        track_vector_unrot = np.array([track[i].x * scale, track[i].y * scale])
        track_vector = np.dot(rot_matrix, track_vector_unrot)
        end_point = np.add(starting_point, track_vector)
        if current_track == i:
            for a in range(0, i, 1): 
                elapsed_rail_ties += track[a].ties
            current_rail_elapsed_ties = train_position - elapsed_rail_ties

            current_rail_ties = track[current_track].ties
            train_point = starting_point + (track_vector / current_rail_ties * current_rail_elapsed_ties)

        starting_point = end_point
    return train_point

def find_closet_track(screen, starting_point, point, track, scale):
    angle = 180
    rail_center_points = [0] * len(track)
    distances = [0] * len(track)
    
    for i in range(0, len(track)):
        rot_angle = math.radians(angle + track[i].angle * track[i].direction)
        rot_matrix = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        angle = math.degrees(rot_angle)

        track_vector_unrot = np.array([track[i].x * scale, track[i].y * scale])
        track_vector = np.dot(rot_matrix, track_vector_unrot)
        end_point = np.add(starting_point, track_vector)

        rail_center_point = starting_point + track_vector / 2
        starting_point = end_point
        rail_center_points[i] = rail_center_point
        distances[i] = np.linalg.norm(rail_center_point - point)

    pygame.draw.circle(screen, (0, 0, 255), rail_center_points[np.argmin(distances)], 20, 0)

def main():
    model = Model.load('train_model_test.model')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_ip = socket.getaddrinfo("0.0.0.0", 9000, socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_ip[0][4])
    sock.setblocking(0)
    print(server_ip[0][4])
    
    scale = 200 # pixels / meter
    starting_point = np.array([400, 200])
    
    straight30 = rail(length = .30, ties = 1)
    straight30_color = rail(length = .30, ties = 1)
    straight30_start = rail(length = .30, ties = 1)

    left30 = rail(radius = .6, angle = 30, direction = -1, ties = 1)
    left30_color = rail(radius = .6, angle = 30, direction = -1, ties = 1)

    right30 = rail(radius = .6, angle = 30, direction = 1, ties = 1)
    right30_color = rail(radius = .6, angle = 30, direction = 1, ties = 1)

    switch = rail(length = .30, ties = 0)

    track = [straight30, straight30, straight30, left30, left30, left30, left30, 
        left30, left30, right30, right30, right30, left30, left30, left30, left30, 
        left30, left30, straight30, straight30, straight30, straight30, left30, left30, left30, straight30]
    train_point = None
    prev_train_point = None
    tie_count = 0
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    
    position1 = []
    position2 = []
    position3 = []
    position4 = []
    position5 = []
    position6 = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        addr, value = read_packet(sock)
        if addr == None:
            pass
        if addr == "/tie_count":
            tie_count = value[0]
            #print(addr)
        if addr == "/color":
            print(value[0], value[1], value[2])
        if addr == "/data":
            position = value[0]
            distance_left_norm = value[1]
            distance_right_norms = value[2]
            magnetox_norm = value[3]
            magnetoy_norm = value[4]
            magnetoz_norm = value[5]
            data_array=[distance_left_norm, distance_right_norms, magnetox_norm, magnetoy_norm, magnetoz_norm]
            print(data_array)
            if position == 1:
                position1.append(data_array)
                np.save("data/position1.npy", position1)
            elif position == 2:
                position2.append(data_array)
                np.save("data/position2.npy", position2)
            elif position == 3:
                position3.append(data_array)
                np.save("data/position3.npy", position3)
            elif position == 4:
                position4.append(data_array)
                np.save("data/position4.npy", position4)
            elif position == 5:
                position5.append(data_array)
                np.save("data/position5.npy", position5)
            elif position == 6:
                position6.append(data_array)
                np.save("data/position6.npy", position6)
            
        if addr == "/sensors":
            distance_left_norm = value[0]
            distance_right_norms = value[1]
            magnetox_norm = value[2]
            magnetoy_norm = value[3]
            magnetoz_norm = value[4]
            data_array=[distance_left_norm, distance_right_norms, magnetox_norm, magnetoy_norm, magnetoz_norm]

            confidences = model.predict(data_array)

            # Get prediction instead of confidence levels
            predictions = model.output_layer_activation.predictions(confidences)

            if(np.max(confidences) > 0.97):
                if predictions[0] + 1 == 1:
                    tie_count = 1
                elif predictions[0] + 1 == 2:
                    tie_count = 7
                elif predictions[0] + 1 == 3:
                    tie_count = 11
                elif predictions[0] + 1 == 4:
                    tie_count = 13
                elif predictions[0] + 1 == 5:
                    tie_count = 19
                elif predictions[0] + 1 == 6:
                    tie_count = 24
            print(tie_count)
                


        screen.fill((0,0,0))
        draw_track(screen, starting_point, track, scale)
        train_point = train_position(starting_point, track, tie_count, scale)
        if train_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), train_point, 10, 0)
            prev_train_point = train_point
        elif prev_train_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), prev_train_point, 10, 0)

        mouse_pos = pygame.mouse.get_pos()
        buttons_pressed = sum(pygame.mouse.get_pressed(num_buttons=3))
        if buttons_pressed > 0:
            pygame.draw.circle(screen, (0, 0, 255), mouse_pos, 20, 0)
            find_closet_track(screen, starting_point, mouse_pos, track, scale)
            
        #pygame.draw.circle(screen, (red, green, blue), (500, 500), 200, 0)
        pygame.display.flip()

    pygame.quit

if __name__ == "__main__":  
    main()

