# Motion Detection

This is for getting sections of the video with motion in it.
You can select the area you want to detect motion.

Created because I sometimes leave my camera on a tripod to film myself doing snowboard tricks and I don't want to go through hours of footage

## Setup
- Use pip to install the requirements in requirements.txt- 
- You can set up the default directory by passing in the directory location into the parameter in main.py
## Execution
- Run the python application
- Select the folder containing the videos you want to process, as most cameras will split long recording into several files
- A preview screen with a frame from the video will appear. Select the area you want to observe by dragging the mouse which will display a rectangle
- If multiple rectangles are draws, the last one will be used
- Press Enter to start
- Will output the clips in the out/{filename} folder