import pygame
import math
import numpy as np
from uosc.server import parse_message
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = socket.getaddrinfo("0.0.0.0", 9000, socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(server_ip[0][4])
print(server_ip[0][4])

class rail:
    def __init__(self, length = 0.0, radius = 0.0, angle = 0.0, direction = 0, ties = 0):
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


def read_packet():
    try:
        pkt = sock.recv(200)
        if not pkt:
            return None, None
        else:
            addr, tags, value = parse_message(pkt)
            return addr, value
    except:
        return None, None

def draw_track(screen, starting_point, track):
    angle = 0
    
    for i in range(0, len(track)):
        rot_angle = math.radians(angle + track[i].angle * track[i].direction)
        rot_matrix = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        angle = math.degrees(rot_angle)

        track_vector_unrot = np.array([track[i].x * 100, track[i].y * 100])
        track_vector = np.dot(rot_matrix, track_vector_unrot)
        end_point = np.add(starting_point, track_vector)
        if track[i].length == 0:
            pygame.draw.line(screen, (255, 255, 255), starting_point, end_point)
        else:
            pygame.draw.line(screen, (255, 255, 0), starting_point, end_point)
    
        starting_point = end_point

def draw_train(screen, starting_point, track, tie_count): 
    angle = 0
    track_length = 0
    track_length_temp = 0
    current_track = 0
    train_position = 0

    for i in range(0, len(track)):
        track_length += track[i].ties
        if track_length > train_position:
            current_track = i
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

        track_vector_unrot = np.array([track[i].x * 100, track[i].y * 100])
        track_vector = np.dot(rot_matrix, track_vector_unrot)
        end_point = np.add(starting_point, track_vector)
        train_point = starting_point + (track_vector / 11 *(train_position % track[current_track].ties))

        if current_track == i:
            screen.fill((0,0,0))
            pygame.draw.circle(screen, (255, 0, 255), train_point, 10, 0)


    
        starting_point = end_point

    
def main():
    starting_point = np.array([400, 400])
    
    straight30 = rail(length = .30, ties = 11)
    left30 = rail(radius = .6, angle = 30, direction = -1, ties = 11)
    right30 = rail(radius = .6, angle = 30, direction = 1, ties = 11)

    #track = [straight30, left30, left30, left30, left30, left30, left30, straight30, left30, left30, left30, left30, left30, left30]
    #track = [straight30, straight30, left30, straight30, right30, straight30]

    track = [straight30, straight30, left30, straight30, right30, straight30, straight30, straight30, straight30, straight30, left30, left30, left30, left30, left30, left30, straight30, straight30, straight30]
    

    tie_count = 0

    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        addr, value = read_packet()
        if addr == None:
            pass
        if addr == "/tie_count":
            tie_count = value[0]
            draw_train(screen, starting_point, track, tie_count)
        
        draw_track(screen, starting_point, track)
        

        pygame.display.flip()

    pygame.quit

if __name__ == "__main__":  
    main()

