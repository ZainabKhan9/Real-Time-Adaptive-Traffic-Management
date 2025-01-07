

import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5 import QtCore

DefaultTime = [11, 11, 11, 11, 11, 11, 11, 11]

class MyApp(QtWidgets.QDialog):
    def __init__(self):
        super(MyApp, self).__init__()
        # Load the UI file (created with Qt Designer)
        uic.loadUi('d1.ui', self)
        
        self.comboBox = self.findChild(QtWidgets.QComboBox, 'comboBox')
        self.imageLabel = self.findChild(QtWidgets.QLabel, 'imageLabel') 

        self.comboBox.currentIndexChanged.connect(self.update_image)
        self.pushButton.clicked.connect(self.ButtonClick)
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
        self.lcd_values_Default = [11, 11, 11, 11, 11, 11, 11, 11]  # Adjust as per the total number of LCDs shown       
        self.lcd_values =  [11, 11, 11, 11, 11, 11, 11, 11]  # Adjust as per the total number of LCDs shown
        
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

    def ButtonClick(self):
        self.timer.stop()

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
        self.lcd_values_Default[LastIndex] = 15
        self.lcd_values[LastIndex] = 15
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

        self.lcdNumber_L.display(self.lcd_values_Default[0])
        self.lcdNumber_LT.display(self.lcd_values_Default[1])
        self.lcdNumber_T.display(self.lcd_values_Default[2])
        self.lcdNumber_RT.display(self.lcd_values_Default[3])
        self.lcdNumber_R.display(self.lcd_values_Default[4])
        self.lcdNumber_RB.display(self.lcd_values_Default[5])
        self.lcdNumber_B.display(self.lcd_values_Default[6])
        self.lcdNumber_LB.display(self.lcd_values_Default[7])

        # Check if we have reached the countdown limit for the current LCD
        if self.lcd_values[self.lcd_index] > 0:
            # Display the current LCD value
            self.display_lcd(self.lcd_index)

            # Decrement the value for the current LCD
            self.lcd_values[self.lcd_index] -= 1
        else:

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
            # Display the current LCD value
            self.display_lcd(self.lcd_index)

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
        
        self.timer.stop()

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

        self.timer.start(1000) 


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


# Run the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())