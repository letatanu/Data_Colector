import pyautogui
from pynput.keyboard import Controller as Controller_Keyboard
import time
from pynput.keyboard import Listener as Listener_Keyboard
from pynput.keyboard import Key, KeyCode
import numpy as np
from win32gui import GetWindowText, GetForegroundWindow
window_name = "World of Warcraft"
def valid_window():
    current_window = (GetWindowText(GetForegroundWindow()))
    desired_window_name = window_name  # Whatever the name of your window should be
    if current_window != desired_window_name:
        return False
    return True
exitKey = ''
def on_press(key):
    global s
    global exitKey
    try:
        if key == Key.enter:
            exitKey = "Enter"
        if key == KeyCode.from_char('s'):
            run()
    except AttributeError:
        raise Exception
import random
def random_float(low, high):
    return random.random()*(high-low) + low
def run():
    count = 0
    width_screen, height_screen = pyautogui.size()
    global exitKey
    while True:
        pyautogui.press('tab')
        time.sleep(random_float(0.3, 0.5))
        pyautogui.press('\'')

        time.sleep(random_float(0.3,0.5))
        pyautogui.press('3')
        time.sleep(random_float(0.3,0.5))
        # for i in range(10):
        # print(rand[0])
        pyautogui.press('2')
        time.sleep(random_float(0.3,0.5))
        # pyautogui.press('4')
        # time.sleep(random_float(0.5, 0.75))
        # pyautogui.press('6')
        # time.sleep(random_float(0.5, 0.75))   '
        pyautogui.press('\/')
        time.sleep(random_float(0.3, 0.5))
        if count % 7  == 0:
            pyautogui.press('5')
            time.sleep(random_float(0.3, 0.75))
            pyautogui.click(width_screen/2-random_float(-50,50), height_screen/2-random_float(50,80))
            time.sleep(random_float(0.3, 0.5))
            pyautogui.press(';')
            time.sleep(random_float(2, 2.5))
            pyautogui.press('=')
            time.sleep(random_float(0.3, 0.5))
            pyautogui.keyDown('d')
            time.sleep(random_float(2, 3))
            pyautogui.keyUp('d')
            count = 0
        count +=1
        if exitKey == "Enter":
            exitKey = ""
            break



with Listener_Keyboard(on_press=on_press) as listener:
    try:
        listener.join()
    except Exception:
        print("Stopped")