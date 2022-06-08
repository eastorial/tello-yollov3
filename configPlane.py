from djitellopy import Tello
import time

def plane(tello, type, value):
    
    time.sleep(1)
    print(type + ' : ' + str(value))
    
    if type == 'move_forward':
        tello.move_forward(value)
    elif type == 'move_back':
        tello.move_back(value)
    elif type == 'move_left':
        tello.move_left(value)
    elif type == 'move_right':
        tello.move_right(value)
    elif type == 'rotate_clockwise':
        tello.rotate_clockwise(value)
    elif type == 'rotate_counter_clockwise':
        tello.rotate_counter_clockwise(value)
    elif type == 'move_up':
        tello.move_up(value)
    elif type == 'move_down':
        tello.move_down(value)
    elif type == 'land':
        tello.land()
    elif type == 'streamoff':
        tello.streamoff()
        
    time.sleep(1)