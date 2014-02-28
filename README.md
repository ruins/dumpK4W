# Summary
Kinect for Windows v2 (ToF sensor) raw data dumper. Takes ~150MB/s of depth, infrared and color frames from the sensor to RAM for a fixed number of frames. It then dumps frames in RAM to HDD. Note that the program is multi-threaded (1 thread per sensor). 

# Hardware platforms tested to run real time: 
*   i5 2500k @ 4.2GHz + GTX 680
*   Intel Haswell i5 NUC (D54250WYK)

# Usage
This is a command line utility and designed to be run from there. It can potentially use a lot of your RAM and HDD space, so take care when running it!

## Example usage: (-h for help to see other options)
dumpK4W.exe -p "C:/path/to/save/data"


# Note
*   You will need OpenCV and Kinect 4 Windows v2 SDK to compile the code
*   You will also need the VC11 redistributable package from MS if you only want to run the binary and not compile the code

# Disclaimer
This is based on preliminary software and/or hardware, subject to change.
