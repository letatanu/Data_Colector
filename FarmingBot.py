import pyautogui
from Learning_Model import Net
from pynput.keyboard import Listener as Listener_Keyboard
from pynput.keyboard import Key
import cv2
from win32gui import GetWindowText, GetForegroundWindow
import numpy as np
window_name = "World of Warcraft"
import torch
import math
import time
import threading
width_screenShot = 639
height_screenShot = 613
off_height = 200
processing = False
c = threading.Condition()
isListening = True
key_dict = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', ';', '\'', '\\']
width_screen, height_screen = pyautogui.size()
desired_window_name = "World of Warcraft"
def convertToAction(number):
    number = 0 if number < 0 else math.floor(number)
    mouseClickY = math.floor(number /10000)
    number = number%10000
    mouseClickX = math.floor(number/100)
    number = number%100
    keyboard = key_dict[number%len(key_dict)]
    return keyboard, (mouseClickX, mouseClickY)

def screen_capture(width, height):
    left = int((width_screen - width) / 2)
    top = int((height_screen - height - off_height) / 2)
    # monitor = {"top": top, "left": left, "width": width, "height": height}
    monitor =(left, top , width, height)
    # grab the data:
    # image capture
    with pyautogui.screenshot(region=monitor) as sct:
        img = np.array(sct)[:,:,:3]
        # cv2.imwrite("./Data/testing/img.png", img)
        scale_percent = 60  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        image = np.array(resized, dtype=np.float) / 255.0
        image = image.transpose((2, 0, 1))
        return image

def runModel():
    global processing
    if not processing: return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    model.load_state_dict(torch.load("./model_test1"))
    model.eval()
    while (processing):
        time.sleep(1)
        if (GetWindowText(GetForegroundWindow())) == desired_window_name:
            img = screen_capture(width_screenShot, height_screenShot)
            input = torch.from_numpy(img).unsqueeze(0).to(device, dtype=torch.float)
            output = (model(input).data).cpu().numpy()
            keyboard, (mouseClickX, mouseClickY) = convertToAction(output.flat[0])
            print(keyboard, mouseClickX, mouseClickY)
            pyautogui.press(keyboard)
            if mouseClickX > 20 or  mouseClickY > 20:
                pyautogui.click(mouseClickX*100/width_screen, mouseClickY*100/height_screen)


def onPress(key):
    global processing, isListening
    if key == Key.enter:
        processing = not processing
    elif key == Key.esc:
        isListening = False
    return isListening
def main():
    with Listener_Keyboard(on_press=onPress) as keyboardListener:
        while isListening:
            t = threading.Thread(target=runModel)
            t.start()
            t.join()
        keyboardListener.join()



if __name__ == '__main__':
    main()