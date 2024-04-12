from Adafruit_PCA9685 import PCA9685
import RPi.GPIO as GPIO
import numpy as np


class DCMotor:
    def __init__(self):
        self.Motor_A_EN = 17
        self.Motor_A_Pin1 = 27
        self.Motor_A_Pin2 = 18
        self.pwm_A = None
        self.HERTZ = 50

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
        if speed < 0:  # 'backward':
            GPIO.output(self.Motor_A_Pin1, GPIO.LOW)
            GPIO.output(self.Motor_A_Pin2, GPIO.HIGH)
            speed = -speed
        elif speed > 0:  # 'forward':
            GPIO.output(self.Motor_A_Pin1, GPIO.HIGH)
            GPIO.output(self.Motor_A_Pin2, GPIO.LOW)
        else:
            self.motorStop()
            return

        if speed > 100:
            speed = 100
        elif speed < 0:
            speed = 0
        self.pwm_A.ChangeDutyCycle(speed)


class ServoMotor:
    def __init__(self):
        self.pwm = PCA9685()
        self.HERTZ = 50
        self.RIGHT_MAX = 230
        self.LEFT_MAX = 370
        self.CENTER = (self.LEFT_MAX + self.RIGHT_MAX) // 2
        self.MAX_ANGLE = np.pi / 2  # 90 degrees in radians

    def setup(self):
        self.pwm.set_pwm_freq(self.HERTZ)
        self.pwm.set_pwm(3, 0, self.CENTER)

    def angle_control(self, angle):
        # angle: -pi/2 ~ pi/2 (in radians)
        angle = np.clip(angle, -self.MAX_ANGLE, self.MAX_ANGLE)
        servo_tick = int((angle + self.MAX_ANGLE) * (self.LEFT_MAX - self.RIGHT_MAX) / (2 * self.MAX_ANGLE) + self.RIGHT_MAX)
        self.pwm.set_pwm(3, 0, servo_tick)