import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5 import QtCore
# from Final_Backend import *
import threading
# DefaultTime = [11, 12, 13, 14, 15, 16, 17, 17]
import numpy as np
import torch
import cv2
import warnings

import pandas as pd
import sklearn
import pickle
import time

# Load the model and scalers
with open('svr_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler_X.pkl', 'rb') as scaler_X_file:
    loaded_scaler_X = pickle.load(scaler_X_file)

with open('scaler_y.pkl', 'rb') as scaler_y_file:
    loaded_scaler_y = pickle.load(scaler_y_file)


# Suppress specific FutureWarning for context manager
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*autocast.*is deprecated.*")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



# Mapping COCO classes to custom vehicle types
vehicle_class_mapping = {
    'car': 'LV',
    'motorcycle': '2_wheeler',
    'bus': 'HV',
    'truck': 'HV'
}

# Box colors for each custom vehicle type
box_colors = {
    'LV': (255, 0, 0),        # Blue for Light Vehicles
    '2_wheeler': (0, 255, 0), # Green for 2 Wheelers
    'HV': (0, 0, 255)         # Red for Heavy Vehicles
}

cap = None 
Loadpath = True
output_size = None
        # Desired output size for display
frame_width = 1080 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = 720 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (frame_width, frame_height)
Calculated_time_By_Model = 0

#_------------------------------------------
# from utils import visualize
from shapely.geometry import Point, Polygon
LeftBottomPoint=(50,680)
LeftTopPoint=(300,500)
RightBottomPoint=(700,680)
RightTopPoint=(800,500)



print("LeftBottomPoint",LeftBottomPoint)
print("LeftTopPoint",LeftTopPoint)
print("RightBottomPoint",RightBottomPoint)
print("RightTopPoint",RightTopPoint)

# Create Point objects
LB = Point(LeftBottomPoint)
LT = Point(LeftTopPoint)
RB = Point(RightBottomPoint)
RT = Point(RightTopPoint)

# Create a Polygon
coords = [LeftBottomPoint,LeftTopPoint,RightTopPoint,RightBottomPoint]
poly = Polygon(coords)
print("p1",LT.within(poly))
print("p2",LB.within(poly))


def Process_Camera(vid_path,duration):
    global Loadpath, cap, output_size, frame_width, frame_height, Calculated_time_By_Model


    if Loadpath == True:
        # Load video
        video_path = 'vid/'+ vid_path + '.mp4'
        print('vid path : ', video_path)
        cap = cv2.VideoCapture(video_path)
        cv2.destroyAllWindows()

        # print('Setting path')
        Loadpath = False
        return True
    else:
        # print('CV part')

        ret, frame = cap.read()
        if not ret:
            # print('frame not found')
            Loadpath = True
            return None 
        
        frame = cv2.resize(frame, output_size)
        # Convert BGR image to RGB and run detection
        results = model(frame[..., ::-1])

        # Get predictions
        predictions = results.xyxy[0]

        # Initialize a dictionary to count vehicle types for the current frame
        frame_vehicle_count = {'LV': 0, '2_wheeler': 0, 'HV': 0}

        # Count detected vehicles in the current frame
        for *box, conf, cls in predictions:
            class_name = model.names[int(cls)]
            if class_name in vehicle_class_mapping:
                category = vehicle_class_mapping[class_name]
                
                # frame_vehicle_count[category] += 1
                # Draw bounding box with the appropriate color
                x1, y1, x2, y2 = map(int, box)

                # Get the center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = Point(center_x, center_y)

                # Check if the center point is within the polygon
                if center_point.within(poly):
                    # Only count this vehicle if it's inside the polygon
                    frame_vehicle_count[category] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colors[category], 2)
                    label = f"{category}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[category], 2)

        # Calculate the total number of vehicles detected in the current frame
        total_vehicles = sum(frame_vehicle_count.values())

        # Display the count of vehicles on the top right corner
        offset_y = 30
        for i, (vehicle_type, count) in enumerate(frame_vehicle_count.items()):
            text = f"{vehicle_type}: {count}"
            cv2.putText(frame, text, (frame_width - 250, 30 + i * offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        ModelInput= [list(frame_vehicle_count.values())]
        # Scale the new data
        new_data_scaled = loaded_scaler_X.transform(ModelInput)
        predicted_scaled = loaded_model.predict(new_data_scaled)

        # Inverse transform the prediction
        predicted_total_time = loaded_scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
        Calculated_time_By_Model = int(predicted_total_time[0])
        # Calculated_time_By_Model = 15 #>>>>>>>>>>>>>>>>>>>
        # Output the predicted value
        print(f'Predicted total time for input {ModelInput}: {predicted_total_time[0]}')
        Time_calculated=f"Time : " + str(int(predicted_total_time[0]))

        total_text = f"Total: {total_vehicles}"
        cv2.putText(frame, total_text, (frame_width - 250, 30 + len(frame_vehicle_count) * offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, Time_calculated, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output video
        # out.write(frame)

        overlay = frame.copy()
            # Green color in BGR
        Red = (0, 0, 255)


            # pts = [100,150,100,200]
            # cv2.polylines(frame, np.array([pts]), True, Red, 5)
        cv2.fillPoly(overlay, np.array([coords]), Red)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


        cv2.imshow(vid_path + '_Time:'+str(duration), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None




def run_for_time(duration,vid):
    global cap
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret = Process_Camera(vid,duration)
        # if ret == None:
        #     break
        # time.sleep(1)  # Optional: Adjust the sleep time as needed
    cap.release()
    cv2.destroyAllWindows()
    # return Counter





class MyApp(QtWidgets.QDialog):
    def __init__(self):
        super(MyApp, self).__init__()
        # Load the UI file (created with Qt Designer)
        uic.loadUi('d1.ui', self)

        self.just_started=True
        
        self.comboBox = self.findChild(QtWidgets.QComboBox, 'comboBox')
        self.imageLabel = self.findChild(QtWidgets.QLabel, 'imageLabel') 

        self.comboBox.currentIndexChanged.connect(self.update_image)
        # self.pushButton.clicked.connect(self.ButtonClick)
        # pushButton

        self.lcdNumber_L = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_L')
        self.lcdNumber_LT = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_LT')
        self.lcdNumber_T = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_T')
        self.lcdNumber_R = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_R')
        self.lcdNumber_RT = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_RT')
        self.lcdNumber_RB = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_RB')        
        self.lcdNumber_B = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_B')
        self.lcdNumber_LB = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber_LB')

        # Countdown values for each LCD
        self.lcd_values_Default = [11, 12, 13, 14, 15, 16, 17, 17]  # Adjust as per the total number of LCDs shown       
        self.lcd_values =  [11, 12, 13, 14, 15, 16, 17, 17]  # Adjust as per the total number of LCDs shown
        
        self.lcdNumber_L.display(self.lcd_values[0])
        self.lcdNumber_LT.display(self.lcd_values[1])
        self.lcdNumber_T.display(self.lcd_values[2])
        self.lcdNumber_RT.display(self.lcd_values[3])
        self.lcdNumber_R.display(self.lcd_values[4])
        self.lcdNumber_RB.display(self.lcd_values[5])
        self.lcdNumber_B.display(self.lcd_values[6])
        self.lcdNumber_LB.display(self.lcd_values[7]) 

        self.lcd_index = 0  # Keep track of which LCD to update

        # Set up a timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_lcd)  
        self.timer.start(1000)  # Start timer with 1-second intervals

        self.show()

    # def ButtonClick(self):
    #             # Set up a timer
    #     self.timer = QtCore.QTimer(self)
    #     self.timer.timeout.connect(self.update_lcd)  
    #     self.timer.start(1000)  # Start timer with 1-second intervals

    
    def Get_model_time(self):
        global Calculated_time_By_Model
        self.timer.stop()
        # print("Button Click Event")
        print( 'self.lcd_index :', self.lcd_index)
        print( ' len(self.lcd_values) :', len(self.lcd_values))

        LastIndex = self.lcd_index + 1
        if(LastIndex == len(self.lcd_values)):
            LastIndex=0
        else:
            while not self.get_lcd_visibility(LastIndex):
                    print('No visitble :' , LastIndex)
                    LastIndex = (LastIndex + 1) % len(self.lcd_values)

        print('Updating Timer for :',LastIndex )
        if int(Calculated_time_By_Model) <= 0:
            Calculated_time_By_Model = 15
        print('updated time : ', int(Calculated_time_By_Model))

        self.lcd_values_Default[LastIndex] = int(Calculated_time_By_Model)
        self.lcd_values[LastIndex] = int(Calculated_time_By_Model)
        self.timer.start(1000) 


    def update_lcd(self):
        # First hide all LCDs
        self.lcdNumber_L.setStyleSheet("color: red;")
        self.lcdNumber_LT.setStyleSheet("color: red;")
        self.lcdNumber_T.setStyleSheet("color: red;")
        self.lcdNumber_RT.setStyleSheet("color: red;")
        self.lcdNumber_R.setStyleSheet("color: red;")
        self.lcdNumber_RB.setStyleSheet("color: red;")
        self.lcdNumber_B.setStyleSheet("color: red;")
        self.lcdNumber_LB.setStyleSheet("color: red;")

        # Reset all LCD displays
        print(' self.lcd_values_Default :' , self.lcd_values_Default)
        # print( ' self.lcd_values[self.lcd_index] : ', self.lcd_values[self.lcd_index] )

        self.lcdNumber_L.display(self.lcd_values_Default[0])
        self.lcdNumber_LT.display(self.lcd_values_Default[1])
        self.lcdNumber_T.display(self.lcd_values_Default[2])
        self.lcdNumber_RT.display(self.lcd_values_Default[3])
        self.lcdNumber_R.display(self.lcd_values_Default[4])
        self.lcdNumber_RB.display(self.lcd_values_Default[5])
        self.lcdNumber_B.display(self.lcd_values_Default[6])
        self.lcdNumber_LB.display(self.lcd_values_Default[7])

        # Check if we have reached the countdown limit for the current LCD
        if self.lcd_values[self.lcd_index] > 0:# and self.just_started == False:
            # Display the current LCD value
            self.display_lcd(self.lcd_index)

            # Decrement the value for the current LCD
            self.lcd_values[self.lcd_index] -= 1
            # self.just_started == False
            # cv2.destroyAllWindows()
        else:
            
            self.Get_model_time()            
            self.lcd_values[self.lcd_index] = self.lcd_values_Default[self.lcd_index]
            # Move to the next visible LCD          
            # self.display_lcd(self.lcd_index)
            # self.lcd_values[self.lcd_index] = self.lcd_values_Default[self.lcd_index]
            # self.display_lcd(self.lcd_index)
            
            self.lcd_index = (self.lcd_index + 1) % len(self.lcd_values)  # Move to the next LCD
        
            # Loop until we find a visible LCD or wrap around
            while not self.get_lcd_visibility(self.lcd_index):
                self.lcd_index = (self.lcd_index + 1) % len(self.lcd_values)

            # Reset or set to desired starting value for the found LCD
            self.lcd_values[self.lcd_index] = self.lcd_values_Default[self.lcd_index]  # Reset to 10 or any preferred value
            
            print( ' self.lcd_index :' ,self.lcd_index)
            print( ' self.lcd_values[self.lcd_index] : ', self.lcd_values[self.lcd_index] )
            

            # if(self.lcd_index + 1 > 7):

            #     ThredTime= self.lcd_values[0]
            # else:
            loc = self.lcd_index+1
            if loc > 7:
                loc=0
            ThredTime= self.lcd_values[loc] - 3  #End before 3 sec.
            print('ThredTime : ', ThredTime)
            global Loadpath
            Loadpath = True
            # if thread

            thread = threading.Thread(target=run_for_time, args=(7,str(self.lcd_index)))
            thread.daemon = True  # Allow thread to be killed when main program exits
            thread.start()
        
            # self.Get_model_time()

            # start_tread_for_x_time
           

    def get_lcd_visibility(self, index):
        """Check if the LCD at the given index is visible."""
        
        if index == 0:
            return self.lcdNumber_L.isVisible()
        elif index == 1:
            return self.lcdNumber_LT.isVisible()
        elif index == 2:
            return self.lcdNumber_T.isVisible()
        elif index == 3:
            return self.lcdNumber_RT.isVisible()
        elif index == 4:
            return self.lcdNumber_R.isVisible()
        elif index == 5:
            return self.lcdNumber_RB.isVisible()
        elif index == 6:
            return self.lcdNumber_B.isVisible()
        elif index == 7:
            return self.lcdNumber_LB.isVisible()
        
        return False  # Default case


    def display_lcd(self, index):
        # This function shows the LCD and sets its value


        if index == 0:
            self.lcdNumber_L.setStyleSheet("color: green;")    
            self.lcdNumber_L.display(self.lcd_values[index])
        elif index == 1:
            self.lcdNumber_LT.setStyleSheet("color: green;") 
            self.lcdNumber_LT.display(self.lcd_values[index])

        elif index == 2:
            self.lcdNumber_T.setStyleSheet("color: green;") 
            self.lcdNumber_T.display(self.lcd_values[index])
        elif index == 3:
            self.lcdNumber_RT.setStyleSheet("color: green;") 
            self.lcdNumber_RT.display(self.lcd_values[index])        

        elif index == 4:
            self.lcdNumber_R.setStyleSheet("color: green;") 
            self.lcdNumber_R.display(self.lcd_values[index])
        elif index == 5:
            self.lcdNumber_RB.setStyleSheet("color: green;") 
            self.lcdNumber_RB.display(self.lcd_values[index])     

        elif index == 6:
            self.lcdNumber_B.setStyleSheet("color: green;") 
            self.lcdNumber_B.display(self.lcd_values[index])
        elif index == 7:
            self.lcdNumber_LB.setStyleSheet("color: green;") 
            self.lcdNumber_LB.display(self.lcd_values[index])

    def update_image(self):
        current_selection = self.comboBox.currentText()
        print(current_selection)
        # try:
        #     self.timer.stop()
        # except:
        #     print('Unable to stop timer')
        # Reset countdown values and index every time the selection changes
        self.lcd_values = self.lcd_values_Default.copy()  # Create a copy of the default values
        self.lcd_index = 0  # Reset index
        
        # Hide all LCDs initially
        self.lcdNumber_L.hide()
        self.lcdNumber_R.hide()
        self.lcdNumber_T.hide()
        self.lcdNumber_B.hide()
        self.lcdNumber_LT.hide()
        self.lcdNumber_LB.hide()
        self.lcdNumber_RT.hide()
        self.lcdNumber_RB.hide()       

        # Use if-elif to determine which LCDs to show
        if current_selection == "Four-way Junction":
            self.lcdNumber_L.show()
            self.lcdNumber_R.show()
            self.lcdNumber_T.show()
            self.lcdNumber_B.show()

        elif current_selection == "Three-way Junction":
            self.lcdNumber_R.show()
            self.lcdNumber_T.show()
            self.lcdNumber_B.show()
            self.lcd_index = 2 # Start from here

        elif current_selection == "Six-way Junction":
            self.lcdNumber_L.show()
            self.lcdNumber_R.show()
            self.lcdNumber_LT.show()
            self.lcdNumber_LB.show()
            self.lcdNumber_RT.show()
            self.lcdNumber_RB.show()

        elif current_selection == "Eight-way Junction":
            self.lcdNumber_L.show()
            self.lcdNumber_R.show()
            self.lcdNumber_T.show()
            self.lcdNumber_B.show()
            self.lcdNumber_LT.show()
            self.lcdNumber_LB.show()
            self.lcdNumber_RT.show()
            self.lcdNumber_RB.show()

        filename = current_selection + '.png'
        pixmap = QtGui.QPixmap(filename)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        self.show()
        # if self.timer.isActive == False :
            # self.timer.start(1000) 


# Run the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())



    # def update_image(self):
    #     current_selection = self.comboBox.currentText()
    #     print(current_selection)
        
    #     self.timer.stop()

    #     # Reset countdown values and index every time the selection changes
    #     self.lcd_values =  self.lcd_values_Default  # Reset values
    #     self.lcd_index = 0  # Reset index
        
    #     # Hide all LCDs initially
    #     self.lcdNumber_L.hide()
    #     self.lcdNumber_R.hide()
    #     self.lcdNumber_T.hide()
    #     self.lcdNumber_B.hide()
    #     self.lcdNumber_LT.hide()
    #     self.lcdNumber_LB.hide()
    #     self.lcdNumber_RT.hide()
    #     self.lcdNumber_RB.hide()       

    #     # Use if-elif to determine which LCDs to show
    #     if current_selection == "Four-way Junction":
    #         self.lcdNumber_L.show()
    #         self.lcdNumber_R.show()
    #         self.lcdNumber_T.show()
    #         self.lcdNumber_B.show()

    #     elif current_selection == "Three-way Junction":
    #         self.lcdNumber_R.show()
    #         self.lcdNumber_T.show()
    #         self.lcdNumber_B.show()
    #         self.lcd_index = 2 # Start from here

    #     elif current_selection == "Six-way Junction":
    #         self.lcdNumber_L.show()
    #         self.lcdNumber_R.show()
    #         self.lcdNumber_LT.show()
    #         self.lcdNumber_LB.show()
    #         self.lcdNumber_RT.show()
    #         self.lcdNumber_RB.show()

    #     elif current_selection == "Eight-way Junction":
    #         self.lcdNumber_L.show()
    #         self.lcdNumber_R.show()
    #         self.lcdNumber_T.show()
    #         self.lcdNumber_B.show()
    #         self.lcdNumber_LT.show()
    #         self.lcdNumber_LB.show()
    #         self.lcdNumber_RT.show()
    #         self.lcdNumber_RB.show()

    #     filename = current_selection + '.png'
    #     pixmap = QtGui.QPixmap(filename)
    #     self.imageLabel.setPixmap(pixmap)
    #     self.imageLabel.setScaledContents(True)

    #     # self.update_lcd()

    #     self.show()

    #     self.timer.start(1000) 


