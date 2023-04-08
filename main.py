print("starting imports")
from uosc.client import Client
from uosc.server import parse_message
from machine import Pin, PWM
import board
import busio
import digitalio
import time
import paa5100ej
import adafruit_vl53l1x
from roboticsmasters_mpu9250 import MPU9250
import time
import math
import network
import socket
import rp2
import wifi
import sys
import uio
import _thread
import gc
print("imports done")
gc.collect()
free_mem = gc.mem_free()
print("Available RAM: {} bytes".format(free_mem))

s_lock = _thread.allocate_lock()

ENA = 0
IN1 = 1
IN2 = 2


left_sensor_power = 13
right_sensor_power = 15

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
        self.right_sensor_power = Pin(right_sensor_power, Pin.OUT)

        self.i2c = board.I2C()
        self.spi = board.SPI()
        self.cs = digitalio.DigitalInOut(board.GP5)
        self.cs.direction = digitalio.Direction.OUTPUT

        self.optical_flow = paa5100ej.PAA5100EJ(self.spi, self.cs)
        self.optical_flow.set_rotation(0)
        self.motiondata = 0, 0
        self.motionx, self.motiony = self.motiondata


        self.right_sensor_power.on()
        self.left_sensor_power.off()
        print(self.i2c.scan())

        time.sleep_ms(200)
        try:
            self.distance_sensor_right = adafruit_vl53l1x.VL53L1X(self.i2c, address=0x30)
        except:
            self.distance_sensor_right = adafruit_vl53l1x.VL53L1X(self.i2c, address=0x29)
            self.distance_sensor_right.set_address(0x30)
            time.sleep_ms(200)
            self.distance_sensor_right = adafruit_vl53l1x.VL53L1X(self.i2c, address=0x30)
        self.left_sensor_power.on()
        time.sleep_ms(200)
        self.distance_sensor_left = adafruit_vl53l1x.VL53L1X(self.i2c, address=0x29)

        self.distance_sensor_left.start_ranging()
        self.distance_sensor_right.start_ranging()
        
        self.distance_sensor_left.distance_mode = 2
        self.distance_sensor_right.distance_mode = 2
        
        self.distance_sensor_left.timing_budget = 33
        self.distance_sensor_right.timing_budget = 33

        print(self.i2c.scan())

        self.magneto_sensor = MPU9250(self.i2c, mpu_addr=0x68)
        self.distance_right = 0
        self.distance_left = 0
        self.last_distance_left = 0
        self.last_distance_right = 0

        self.magnetox = 0
        self.magnetoy = 0
        self.magnetoy = 0

        self.speed = 0
        free_mem = gc.mem_free()
        print("Available RAM: {} bytes".format(free_mem))
        gc.collect()
        free_mem = gc.mem_free()
        print("Available RAM: {} bytes".format(free_mem))
        self.start_read_motion_sensor_thread()

    def start_read_motion_sensor_thread(self):
        time.sleep_ms(200)
        print("attempting to start thread")
        while True:
            try:
                _thread.start_new_thread(self.read_motion_sensor, ())
                break
            except Exception as e:
                print(e)
                print("trying again....")
                gc.collect()
                time.sleep_ms(40)
        print("thread started")
        time.sleep_ms(200)
        
    def read_motion_sensor(self):
        while True:
            temp_motion_data = self.optical_flow.get_motion(5)
            #print("bla")
            s_lock.acquire()
            self.motiondata = temp_motion_data
            s_lock.release()
            
        
    def read_sensors(self):
        self.distance_right = self.distance_sensor_right.distance
        self.distance_left = self.distance_sensor_left.distance
        if self.distance_right == None:
            self.distance_right = 400

        if self.distance_left == None:
            self.distance_left = 400

        self.magneto = self.magneto_sensor.magnetic
        self.gyro = self.magneto_sensor.gyro
        self.acceleration = self.magneto_sensor.acceleration
        if not s_lock.locked():
            s_lock.acquire()
            if self.motiondata is not None:
                self.motionx, self.motiony = self.motiondata
            s_lock.release()

        #print(x,y, dr, raw_sum, raw_max, raw_min, obs)
        self.speed = math.sqrt(self.motionx ** 2 + self.motiony ** 2)

        self.distance_right_norm = (self.distance_right / 400) - 1
        self.distance_left_norm = (self.distance_left / 400) - 1

        self.magnetox_norm = (self.magneto[0] / 128)
        self.magnetoy_norm = (self.magneto[1] / 128)
        self.magnetoz_norm = (self.magneto[2] / 128)

        self.accelx_norm = (self.acceleration[0] / 128)
        self.accely_norm = (self.acceleration[1] / 128)
        self.accelz_norm = (self.acceleration[2] / 128)

        self.gyrox_norm = (self.gyro[0] / 128)
        self.gyroy_norm = (self.gyro[1] / 128)
        self.gyroz_norm = (self.gyro[2] / 128)

        self.speed_norm = (self.speed / 100)

        self.last_distance_right = self.distance_right
        self.last_distance_left = self.distance_left
        
        time.sleep_ms(55)

        return self.distance_right_norm, self.distance_left_norm, \
            self.magnetox_norm, self.magnetoy_norm, self.magnetoz_norm, \
            self.accelx_norm, self.accely_norm, self.accelz_norm, \
            self.gyrox_norm, self.gyroy_norm, self.gyroz_norm, \
            self.speed_norm


    
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
                print(speed)


        sensor_package.read_sensors()
        #print("left: ",sensor_package.distance_left_norm, "right: ",sensor_package.distance_right_norm)
        osc.send('/sensors',sensor_package.distance_right_norm, sensor_package.distance_left_norm, \
            sensor_package.magnetox_norm, sensor_package.magnetoy_norm, sensor_package.magnetoz_norm, \
            sensor_package.accelx_norm, sensor_package.accely_norm, sensor_package.accelz_norm, \
            sensor_package.gyrox_norm, sensor_package.gyroy_norm, sensor_package.gyroz_norm, \
            sensor_package.speed_norm)
        motor.rotate(speed)

if __name__ == "__main__":
    while True:
        try:
            log_file = open("error.log", "w")
            main()
        except Exception as e:
            # Catch the exception and write the error message to the log file
            error_message = "An error occurred: {}\n".format(e)
            log_file.write(error_message)
            # Print the error message with traceback to the console for debugging
            tb_str = uio.StringIO()
            sys.print_exception(e, tb_str)
            traceback_str = tb_str.getvalue()
            sys.stderr.write(traceback_str)
            log_file.write(traceback_str)
        log_file.close()
