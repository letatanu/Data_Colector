import cv2
import numpy as np
from win32gui import GetWindowText, GetForegroundWindow
import pyautogui
from pynput.mouse import Listener as Listener_Keyboard
import time


#Congfig
windowName = "World of Warcraft"
width_screenShot = 639
height_screenShot = 613
off_height = 200


class Fishing():
    def __init__(self, dbName):
        self.dbName = dbName
    def onPress(self, key):
        try:
            keyChar = key.char
            current_window = (GetWindowText(GetForegroundWindow()))
            desired_window_name = self.windowName  # Whatever the name of your window should be
            if keyChar == '1' and current_window == desired_window_name and self.windowName != "":
                time.sleep(1)

        except AttributeError:
            print("Key error")
    def collectDB(self, forWindowName = windowName):
        self.windowName = forWindowName
        self.keyboardListener = Listener_Keyboard(on_press=self.onPress)


