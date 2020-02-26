# Data_Colector
This project tries to train a bot that can play World of Warcraft; for example: using skills efficiently and farming mobs. 

## Collecting Data
The training data is collected from real-time playing: Screen shot + Button pressed.
The pressed button will be the label of that screen shot. 

## Training
Using Pytorch to design a network. See ```Learning_Model``` for more detail. 

## Using the trained model to play WOW
The program will constantly capture a screen shot, then passing that screen shot to the trained model to get the output button. See ```FarmingBot``` for more detail. 
