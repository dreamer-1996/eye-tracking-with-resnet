# CANVAS PARAMTERS



def get_screenWH():
    try:
       import tkinter
       root = tkinter.Tk()
       root.withdraw()
       W_px, H_px = root.winfo_screenwidth(), root.winfo_screenheight()
       #print(WIDTH, HEIGHT)
    except:
       import pyautogui
       W_px, H_px = pyautogui.size()
       #print(WIDTH, HEIGHT)


#------------------------------------------------
# PARAMS OF THE MACBOOK PRO 13 
# H_px = 900 # WITH IN PIX
# W_px = 1440
W_px = 1920.00
H_px = 1080.00

#get_screenWH()
#print(W_px,H_px)
#print(W_px,H_px)
# FOLLOWING PARAMS must be self defined!
H_m = 0.200 #  SCREEN HEIGHT IN M, measured with ruler
W_m = 0.350 #  SCREEN WIDTH IN M, measured with ruler
top_dist = 3.000 # DISTANCE FROM CAMERA SENSOR CENTER TO SCREEN 1cm or 0.8 cm
bottom_line = 120.00
adj_H = H_px - bottom_line
GRID_STEP = 50 # TRY 50
IMG_SCALE = 1 #0.3

# COLORS
#------------------------------------------------
RED = (0,0,255)
WHITE = (255,255,255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# FOR VIDEO
#------------------------------------------------
FPS = 20.0
VID_H = 720
VID_W = 1280
