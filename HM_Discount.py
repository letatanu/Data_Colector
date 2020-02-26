from pynput.keyboard import Controller as Controller_Keyboard
import pyautogui
from pynput.mouse import Button
import time
from pynput.keyboard import Listener as Listener_Keyboard
from pynput.keyboard import Key
from pynput.mouse import Listener as Listener_Mouse
# start_time = 0
# end_time = 0
# from robobrowser import RoboBrowser
import requests
sequence = []
class Mouse_Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y
class MyException(Exception): pass

def scanCode(i):
    if i > 9999 or i < 0:
        return 0
    return '{0:04d}'.format(i)

def recording():
    def on_click(x, y, button, pressed):
        if pressed:
            if button == Button.left:
                # global start_time
                # global end_time
                # if len(sequence) != 0:
                    # end_time = time.time()
                    # sequence.append((end_time - start_time))
                    # start_time = time.time()
                click = Mouse_Position(x, y)
                sequence.append(click)
                # else:
                #     start_time = time.time()
                #     end_time = 0
                #     click = Mouse_Position(x, y)
                #     sequence.append(click)
            else:
                raise MyException(button)



    with Listener_Mouse(on_click=on_click) as  listener_mouse:
        try:
            listener_mouse.join()
        except MyException as e:
            print(e)

    # listener_keyboard = Listener_Keyboard(on_press=on_press)
    #
    #
    # try:
    #     listener_keyboard.start()
    #     listener_mouse.start()
    #     listener_keyboard.join()
    #     listener_mouse.join()
    # except MyException as e:
    #     print('{0} was pressed'.format(e.args[0]))
    #     listener_keyboard.stop()
    #     listener_mouse.stop()

def replay(times, start = 0):
    f = sequence[-2:]
    times = times*2

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            return False
    i = start
    while :
        with Listener_Keyboard(
                on_release=on_release) as listener:
            listener.join()

        job = f[i%2]
        if isinstance(job, Mouse_Position):
            pyautogui.click(job.x,job.y)
            print(job.x, job.y)
            if i % 2:
                time.sleep(3)
            else:
                keyboard = Controller_Keyboard()
                code = scanCode(int(i/2))
                print("code: ", code)
                pyautogui.hotkey("ctrlleft", "a")
                pyautogui.press('backspace')
                for char in code:
                    keyboard.press(char)
        i += 1




def main():
   recording()

   def on_press(key):
       times = 9999
       if key == Key.esc:
           raise MyException(key)
       if key == Key.enter:
           replay(times)
           listener.stop()

   with Listener_Keyboard(on_press=on_press) as listener:
       try:
           listener.join()
       except MyException as E:
           print(E)




if __name__ == '__main__':
    main()