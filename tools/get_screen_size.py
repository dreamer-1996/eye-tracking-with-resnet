
def get_screenWH():
    try:
       import tkinter
       root = tkinter.Tk()
       root.withdraw()
       WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
       print(WIDTH, HEIGHT)
    except:
       import pyautogui
       WIDTH, HEIGHT = pyautogui.size()
       print(WIDTH, HEIGHT)
    #return WIDTH, HEIGHT
#print(resolution)

#WIDTH, HEIGHT = get_screenWH()
get_screenWH()

