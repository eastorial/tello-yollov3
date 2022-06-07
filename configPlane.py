from djitellopy import Tello
import time

def plane(tello, type, value):
    
    print(type + ' : ' + str(value))
    
    if type == 'move_forward':
        time.sleep(3)
        tello.move_forward(value)
    elif type == 'move_back':
        time.sleep(3)
        tello.move_back(value)
    elif type == 'move_left':
        time.sleep(3)
        tello.move_left(value)
    elif type == 'move_right':
        time.sleep(3)
        tello.move_right(value)
    elif type == 'rotate_clockwise':
        time.sleep(3)
        tello.rotate_clockwise(value)
    elif type == 'rotate_counter_clockwise':
        time.sleep(3)
        tello.rotate_counter_clockwise(value)
    elif type == 'move_up':
        time.sleep(3)
        tello.move_up(value)
    elif type == 'move_down':
        time.sleep(3)
        tello.move_down(value)
    elif type == 'land':
        time.sleep(3)
        tello.land()