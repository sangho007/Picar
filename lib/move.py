from PID import PD_Controller
from Adafruit_PCA9685 import PCA9685
import RPi.GPIO as GPIO
import numpy as np

class VehicleMove:
    def __init__(self):
        self.Motor_B_EN = 4
        self.Motor_B_Pin1 = 14
        self.Motor_B_Pin2 = 15
        self.Motor_A_EN = 17
        self.Motor_A_Pin1 = 27
        self.Motor_A_Pin2 = 18
        self.pwm_A = None
        self.pwm = PCA9685()
        self.HERTZ = 50
        self.servo_tick = 300

    def setup(self):
        # Motor initialization
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.Motor_A_EN, GPIO.OUT)
        GPIO.setup(self.Motor_A_Pin1, GPIO.OUT)
        GPIO.setup(self.Motor_A_Pin2, GPIO.OUT)
        self.pwm_A = GPIO.PWM(self.Motor_A_EN, self.HERTZ)
        self.pwm_A.start(0)
        self.motorStop()
        self.pwm.set_pwm_freq(self.HERTZ)
        self.pwm.set_pwm(0, 0, self.servo_tick)
    def motorStop(self):
        # Motor stops
        GPIO.output(self.Motor_A_Pin1, GPIO.LOW)
        GPIO.output(self.Motor_A_Pin2, GPIO.LOW)
        GPIO.output(self.Motor_A_EN, GPIO.LOW)
        self.pwm_A.ChangeDutyCycle(0)

    def destroy(self):
        self.motorStop()
        GPIO.cleanup()

    def move(self, speed):
        # speed = 0~100
        if speed < 0:  #'backward':
            GPIO.output(self.Motor_A_Pin1, GPIO.HIGH)
            GPIO.output(self.Motor_A_Pin2, GPIO.LOW)
        elif speed > 0: #'forward':
            GPIO.output(self.Motor_A_Pin1, GPIO.LOW)
            GPIO.output(self.Motor_A_Pin2, GPIO.HIGH)
        else:
            self.motorStop()
            return
        
        if speed > 100:
            speed = 100
        elif speed < 0:
            speed = 0
        self.pwm_A.ChangeDutyCycle(speed)

    def angle_control(self,servo_tick):
        self.pwm.set_pwm(0, 0, servo_tick)
        
    def yaw_controll(self,error):
        RIGHT_MAX = 230
        LEFT_MAX = 370
        MAX_ERROR = np.pi
        self.servo_tick = (-error + MAX_ERROR) * (LEFT_MAX - RIGHT_MAX) / (MAX_ERROR - (-MAX_ERROR)) + RIGHT_MAX
        
        self.servo_tick = min(max(self.servo_tick, RIGHT_MAX), LEFT_MAX)
        
        self.servo_tick = int(self.servo_tick)
