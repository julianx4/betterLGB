from uosc.client import Client
from uosc.server import parse_message
from machine import Pin, PWM
import time
import network
import socket
import rp2
import wifi


ENA = 0
IN1 = 1
IN2 = 2


wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(wifi.WIFI2[0], wifi.WIFI2[1])

while not wlan.isconnected():
    print("trying to connect to wifi....")
    time.sleep(0.5)
print("\x1b[2J")
print(wlan.ifconfig())

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = socket.getaddrinfo("0.0.0.0", 9001, socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(server_ip[0][4])
sock.setblocking(False)

def wapCreate():
    rp2.country('DE')
    wap = network.WLAN(network.AP_IF)
    wap.config(essid='theTrain', password='train123')
    wap.active(True)
    netConfig = wap.ifconfig()
    print('IPv4-Adresse:', netConfig[0], '/', netConfig[1])
    print('Standard-Gateway:', netConfig[2])
    print('DNS-Server:', netConfig[3])

def read_packet():
    try:
        pkt = sock.recv(200)
        if not pkt:
            print("here")
            return None, None
        else:
            addr, tags, value = parse_message(pkt)
            return addr, value
    except:
        return None, None

class Motor:
    def __init__(self, EN, IN1, IN2, frequency = 500):
        self.IN1 = Pin(IN1, Pin.OUT) 
        self.IN2 = Pin(IN2, Pin.OUT)  
        self.EN = PWM(Pin(EN))  
        self.EN.freq(frequency)
        self.EN.duty_u16(0)
    
    def rotate(self, speed):
        self.speed = speed
        if self.speed < 0:
            self.IN1.on()
            self.IN2.off()
            print("reverse")
        if self.speed > 0:
            self.IN1.off()
            self.IN2.on()
            print("forward")
        if self.speed == 0:
            self.IN1.off()
            self.IN2.off()
        set_speed = int(0 + (65536 - 0)*(abs(speed) / 100))
        self.EN.duty_u16(set_speed)
        #print(set_speed)

class Sensor:
    def __init__(self):
        self.prev_sensor = 0
        self.threshold = 0.5
        self.count = 0

    def RR_tie_detector(self, sensor):
        if sensor < self.threshold:
            if self.prev_sensor > self.threshold:
                self.count += 1
                self.prev_sensor = sensor
        if sensor > self.threshold:
            self.prev_sensor = sensor
        return self.count
        

def main():
    motor = Motor(ENA, IN1, IN2)
    RR_tie_sensor = Sensor()

    osc = Client(('192.168.0.182', 9000))

    speed = 0
    sensor_value = 0

    prev_tie_count = 0

    while True:
        while True:
            addr, value = read_packet()
            if addr == None:
                break
            if addr == "/speed":
                speed = (value[0] - 0.5) * 2 * 100
                if -1 <= speed <= 1:
                    speed = 0
            if addr == "/sensor":
                sensor_value = value[0]

        tie_count = RR_tie_sensor.RR_tie_detector(sensor_value)

        if tie_count is not prev_tie_count:
            osc.send('/tie_count', tie_count)
            prev_tie_count = tie_count
            print(tie_count)


        motor.rotate(speed)

if __name__ == "__main__":
    main()