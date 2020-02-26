from db_handler import DB_Handler
# from pynput.keyboard import Listener, KeyCode




def main():
    # def on_press(key):
    #     listener.stop()
    #     if key == KeyCode.from_char('s'):
    #         db_handler = DB_Handler(data_dir="H:\Data_Collector\Data", scene_dir="test1")
    #         db_handler.colecting_data()
    # with Listener(on_press=on_press) as listener:
    #     listener.join()1


    db_handler = DB_Handler(data_dir="H:\Data_Collector\Data", scene_dir="test1")
    db_handler.colecting_data(db_set="testing", for_window_name="World of Warcraft")
    # db_handler.remove_screenShot("img_1532.jpg", db_set="training")
    # db_handler.create_labels_file()

    ##


if __name__ == '__main__':
    main()