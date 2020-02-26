import os
import pickle
import mss.tools
import pyautogui
from pynput.keyboard import Listener as Listener_Keyboard
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Listener as Listener_Mouse
from pynput.mouse import Button
import cv2
import numpy as np
from win32gui import GetWindowText, GetForegroundWindow
width_screenShot = 639
height_screenShot = 613
off_height = 200
key_dict_ = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', ';', '\'', '\\']


# def convert_label_to_numer(label):
#     first = np.where(key_dict_ == label[0])[0]
#     second = 0 if label[1] == -1 else label[1]
#     third = 0 if label[2] == -1 else label[2]
#     return first + second * 10 + third * 100

class DB_Handler:
    def __init__(self, data_dir, scene_dir):
        self.scene_dir = scene_dir
        self.data_dir = data_dir
        self.stopRunning = False
        self.width_screen, self.height_screen = pyautogui.size()




    def get_labels(self):
        self.labels = []
        for val in self.db.values():
            if val not in self.labels:
                self.labels.append(val)
        return self.labels

    def read_db(self, db_set):
        db_dir = os.path.join(self.data_dir, db_set , self.scene_dir)
        if not os.path.isdir(db_dir):
            os.mkdir(db_dir)
        db_file = os.path.join(db_dir, "db.txt")
        if os.path.isfile(db_file):
            with open(db_file, 'rb') as db_open:
                self.db = pickle.load(db_open)
                for screen_shot in self.db:
                    if not os.path.isfile(os.path.join(db_dir,screen_shot)):
                        del self.db[screen_shot]
        else:
            self.db = {}
            with open(db_file, 'wb') as file:
                pickle.dump(self.db,file)
        return self.db

    def read_config(self, db_set):
        db_dir = os.path.join(self.data_dir, db_set,self.scene_dir)
        if not os.path.isdir(db_dir):
            os.mkdir(db_dir)
        config_file = os.path.join(db_dir, "config.txt")
        if os.path.isfile(config_file):
            # line 1: number of screen shots
            # line 2: screen config: (top, left, width, height)
            with open(config_file, 'rb') as cf_file:
                self.config = pickle.load(cf_file)
        else:
            with open(config_file, 'wb') as file:
                width_screen, height_screen = pyautogui.size()
                left = int((width_screen - width_screenShot) / 2)
                top = int((height_screen - height_screenShot - off_height) / 2)
                config_file = {'screenShot_number': 0,
                               'screen': [top, left, width_screen, height_screen]}
                pickle.dump(config_file, file)
                self.config = config_file

        return self.config


    def writing_db(self, db_file, db_set):
        db_dir = os.path.join(self.data_dir, db_set,self.scene_dir)
        if os.path.isdir(db_dir):
            db_link = os.path.join(db_dir,'db.txt')
            with open(db_link, 'wb') as file:
                pickle.dump(db_file, file)

    def writing_config(self, config_file, db_set):
        db_dir = os.path.join(self.data_dir, db_set,self.scene_dir)
        if os.path.isdir(db_dir):
            config_link = os.path.join(db_dir,"config.txt")
            with open(config_link, "wb") as file:
                pickle.dump(config_file, file)

    def add_new_screenShot(self, sct_img, button_keyboard, db_set):
        ####### Config file
        self.read_config(db_set)
        self.config["screenShot_number"] += 1
        self.writing_config(self.config, db_set=db_set)
        ###### db file
        self.read_db(db_set=db_set)

        screenShot_name = "img_{}.jpg".format(self.config["screenShot_number"])
        img_path = os.path.join(self.data_dir, db_set,self.scene_dir,screenShot_name)
        # png =mss.tools.to_png(sct_img.rgb, sct_img.size)
        img = np.array(sct_img)
        cv2.imwrite(img_path, img)

        def convert_label_to_number(button_keyboard):
            global key_dict_
            try:
                a = button_keyboard.char
            except AttributeError:
                a = button_keyboard
            find = np.where(np.array(key_dict_) == a)
            first = 0 if not len(find[0]) else find[0][0]
            return first
            # second = 0 if mouse_clickX == -1 else mouse_clickX
            # third = 0 if mouse_clickY== -1 else mouse_clickY
            # return first + second * 100 + third * 10000

        self.db[screenShot_name] = convert_label_to_number(button_keyboard)
        self.writing_db(self.db, db_set=db_set)

    def remove_screenShot(self, screenShot_name, db_set):
        #### db file
        self.read_db(db_set=db_set)
        try:
            del self.db[screenShot_name]
            img_path = os.path.join(self.data_dir, db_set, self.scene_dir, screenShot_name)
            os.remove(img_path)

            ### config file
            self.read_config(db_set)
            self.config["screenShot_number"] += 1
            print(self.config)
            print(self.db)
            self.writing_config(self.config, db_set=db_set)
            self.writing_db(self.db, db_set)
        except:
            print(screenShot_name, " does not exist")

    def screen_capture(self, width, height, button_keyboard):
        left = 760
        top = 1300
        monitor = {"top": top, "left": left, "width": 1416-left, "height": 1418-top}
        # grab the data:
        # image capture
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            # save image
            self.add_new_screenShot(sct_img, button_keyboard, db_set = self.db_set)

    def on_press(self, key):
        # check if an input key is valid
        def valid_buttons(key):
            current_window = (GetWindowText(GetForegroundWindow()))
            desired_window_name = self.window_name  # Whatever the name of your window should be
            if current_window != desired_window_name and self.window_name != "":
                return False
            if key == 'nil':
                return False

            try:
                key_ = key.char
                if key_ in key_dict_:
                    return True
            except AttributeError:
                if key in key_dict_:
                    return True
            return False
        ######################

        if valid_buttons(key):
            print("valid")
            self.screen_capture(width=width_screenShot, height=height_screenShot, button_keyboard=key)
        else:
            try:
                if key == Key.esc:
                    self.stopRunning = True
                    # self.listener_mouse.stop()
                    return False
            except AttributeError:
                print("Nothing happens")


    # def on_click(self, x, y, button, pressed):
    #     # check if a click is valid
    #     def valid_click(x, y, button):
    #         current_window = (GetWindowText(GetForegroundWindow()))
    #         desired_window_name = self.window_name  # Whatever the name of your window should be
    #         if current_window != desired_window_name and self.window_name != "":
    #             return False
    #         if x == -1 or y == -1 or button == Button.right:
    #             return False
    #         return True
    #     ####################################
    #     if self.stopRunning:
    #         return False
    #     if pressed and valid_click(x, y, button):
    #         print('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
    #         self.screen_capture(width= width_screenShot, height= height_screenShot, button_keyboard="", mouse_clickX=x, mouse_clickY=y, db_set=self.db_set)


    def colecting_data(self, db_set, for_window_name = "World of Warcraft"):
        self.db_set = db_set
        self.window_name = for_window_name
        ### config file
        self.read_config(db_set=db_set)

        ### db file
        self.read_db(db_set=db_set)
        print("Starting collecting")


        # self.listener_mouse = Listener_Mouse(on_click=self.on_click)
        self.listener_keyboard = Listener_Keyboard(on_press=self.on_press)
        try:
            # self.listener_mouse.start()
            self.listener_keyboard.start()
            # self.listener_mouse.join()
            self.listener_keyboard.join()
        finally:
            self.listener_keyboard.stop()
            # self.listener_mouse.stop()






