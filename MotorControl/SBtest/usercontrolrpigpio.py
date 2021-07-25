import RPi.GPIO as gpio
import time

def init():
  gpio.setmode(gpio.BOARD)
  gpio.setup(7, gpio.OUT)
  gpio.setup(11, gpio.OUT)
  servo1=GPIO.PWM(11,50)
  servo1.start(0)
  print("Wating for 2 seconds")
  time.sleep(2)
  
def forward(tf):
  servo1.ChangeDutyCycle(7)
  gpio.output(7, True)
  time.sleep(tf)
  gpio.cleanup()
 
def left(tf):
  servo1.ChangeDutyCycle(10)
  gpio.output(7, True)
  time.sleep(tf)
  gpio.cleanup()
  
def right(tf):
  servo1.ChangeDutyCycle(4)
  gpio.output(7, True)
  time.sleep(tf)
  gpio.cleanup()
  
def backward(tf):
  servo1.ChangeDutyCycle(7)
  gpio.output(7, False)
  time.sleep(tf)
  gpio.cleanup()
  
  def key_input(event):
    init()
    print 'Key', event.char
    key_press=event.char
    sleep_time=0.030
    print("press user control button")
     if key_press.lower()== 'w':
        forward(sleep_time)
     elif key_press.lower() == 's':
        backward(sleep_time)
     elif key_press.lower() == 'a':
        left(sleep_time)
     elif key_press.lower() == 'd':
        right(sleep_time)  
  
 
  
  command=tk.Tk()
  command.blind('<keyPress>', key_input)
  command.mainloop()
  
  
  
  
