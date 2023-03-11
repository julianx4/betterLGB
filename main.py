from uosc.client import Client
from uosc.server import parse_message
from machine import Pin, PWM, I2C
import VL53L0X
from mpu9250 import MPU9250
import time
import network
import socket
import rp2
import wifi

ENA = 0
IN1 = 1
IN2 = 2

SDA = 10
SCL = 11

left_sensor_power = 13

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
            self.IN1.off()
            self.IN2.on()
        if self.speed > 0:
            self.IN1.on()
            self.IN2.off()
        if self.speed == 0:
            self.IN1.off()
            self.IN2.off()
        set_speed = int(0 + (65536 - 0)*(abs(speed) / 100))
        self.EN.duty_u16(set_speed)

class SensorPackage:
    def __init__(self):
        self.left_sensor_power = Pin(left_sensor_power, Pin.OUT)
        self.left_sensor_power.off()
        self.i2c = I2C(1, scl=Pin(SCL), sda=Pin(SDA))
        print(self.i2c.scan())

        time.sleep_ms(200)
        self.distance_sensor_right = VL53L0X.VL53L0X(self.i2c, address=0x29)
        self.distance_sensor_right.change_address(0x30)
        time.sleep_ms(200)
        self.distance_sensor_right = VL53L0X.VL53L0X(self.i2c, address=0x30)
        self.left_sensor_power.on()
        time.sleep_ms(200)
        self.distance_sensor_left = VL53L0X.VL53L0X(self.i2c, address=0x29)

        #self.distance_sensor_right.set_Vcsel_pulse_period(self.distance_sensor_right.vcsel_period_type[0], 18)
        #self.distance_sensor_right.set_Vcsel_pulse_period(self.distance_sensor_right.vcsel_period_type[1], 14)
        
        #self.distance_sensor_left.set_Vcsel_pulse_period(self.distance_sensor_left.vcsel_period_type[0], 18)
        #self.distance_sensor_left.set_Vcsel_pulse_period(self.distance_sensor_left.vcsel_period_type[1], 14)

        self.magneto_sensor = MPU9250(self.i2c)
        self.distance_right = 0
        self.distance_left = 0
        self.magnetox = 0
        self.magnetoy = 0
        self.magnetoy = 0
        

    def read_sensors(self):
        self.distance_right = min(4000, self.distance_sensor_right.read())
        self.distance_left = min(4000, self.distance_sensor_left.read())

        self.magneto = self.magneto_sensor.magnetic

        self.distance_right_norm = (self.distance_right / 2000) - 1
        self.distance_left_norm = (self.distance_left / 2000) - 1
        self.magnetox_norm = (self.magneto[0] / 128)
        self.magnetoy_norm = (self.magneto[1] / 128)
        self.magnetoz_norm = (self.magneto[2] / 128)

    
def wapCreate():
    rp2.country('DE')
    wap = network.WLAN(network.AP_IF)
    wap.config(essid='theTrain', password='train123')
    wap.active(True)
    netConfig = wap.ifconfig()
    print('IPv4-Adresse:', netConfig[0], '/', netConfig[1])
    print('Standard-Gateway:', netConfig[2])
    print('DNS-Server:', netConfig[3])

def read_packet(sock):
    try:
        pkt = sock.recv(200)
        if not pkt:
            #print("here")
            return None, None
        else:
            addr, tags, value = parse_message(pkt)
            return addr, value
    except:
        return None, None
       
def main():
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

    motor = Motor(ENA, IN1, IN2)
    sensor_package = SensorPackage()

    osc = Client(('192.168.0.255', 9000))

    speed = 0
    position = 0

    while True:
        while True:
            addr, value = read_packet(sock)
            if addr == None:
                break
            if addr == "/speed":
                speed = (value[0] - 0.5) * 2 * 100
                if -1 <= speed <= 1:
                    speed = 0
                #print(speed)
            if addr == "/button1":
                sensor_package.read_sensors()
                osc.send('/data', 1, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

            if addr == "/button2":
                sensor_package.read_sensors()
                osc.send('/data', 2, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

            if addr == "/button3":
                sensor_package.read_sensors()
                osc.send('/data', 3, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

            if addr == "/button4":
                sensor_package.read_sensors()
                osc.send('/data', 4, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

            if addr == "/button5":
                sensor_package.read_sensors()
                osc.send('/data', 5, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

            if addr == "/button6":
                sensor_package.read_sensors()
                osc.send('/data', 6, sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)

        sensor_package.read_sensors()
        osc.send('/sensors',sensor_package.distance_left_norm, sensor_package.distance_right_norm, sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm)
        motor.rotate(speed)

if __name__ == "__main__":
    main()

"""
i2c = I2C(0)
i2c = I2C(0, I2C.MASTER)
i2c = I2C(0, pins=('P10','P9'))
i2c.init(I2C.MASTER, baudrate=9600)

# Create a VL53L0X object
tof = VL53L0X.VL53L0X(i2c)

tof.set_Vcsel_pulse_period(tof.vcsel_period_type[0], 18)

tof.set_Vcsel_pulse_period(tof.vcsel_period_type[1], 14)


while True:
# Start ranging
    tof.start()
    tof.read()
    print(tof.read())
    tof.stop()
"""