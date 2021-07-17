import curses
from gpiozero import Motor
from gpiozero import Servo

servo=Servo(17)
motor=Motor(forward=20, backward=21)

actions = {
    curses.KEY_UP:   
    servo.mid()
    dc_motor.forward(speed=1),
    curses.KEY_DOWN:  
    dc_motor.backward(speed=1),
    curses.KEY_LEFT:  
    servo.min()
    dc_motor.forward(speed=1),
    curses.KEY_RIGHT: 
    servo.max()
    dc_motor.forward(speed=1)
    ,
}

def main(window):
    next_key = None
    while True:
        curses.halfdelay(1)
        if next_key is None:
            key = window.getch()
        else:
            key = next_key
            next_key = None
        if key != -1:
            # KEY PRESSED
            curses.halfdelay(3)
            action = actions.get(key)
            if action is not None:
                action()
            next_key = key
            while next_key == key:
                next_key = window.getch()
            # KEY RELEASED
            robot.stop()

curses.wrapper(main)
