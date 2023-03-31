import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("..\Lab1\material\Video.mp4")

assert video.isOpened(), "Cannot load video"

MOG_VERSION = 2

N_MIXTURES = 5
HISTORY = 200
NOISE_SIGMA = 1
BACKGROUND_RATIO = .1 # low: most things as part of foreground
LEARNING_RATE = -1 # -1: automatic

frames = []
motion = []

# Version selection
if MOG_VERSION == 1:
    bgExtractor = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA); 

elif MOG_VERSION == 2:
    # automatically estimates the correct # of gaussians etc.
    bgExtractor = cv2.createBackgroundSubtractorMOG2()

# Read frame by frame
while(video.isOpened()):
  
  ret, frame = video.read()

  if ret == True:

    # Grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(frame_gray)

    # Version 1: mask
    fgMask = bgExtractor.apply(frame_gray, LEARNING_RATE)

    # Version 2: background
    if MOG_VERSION == 2:
        bg = bgExtractor.getBackgroundImage()
        cv2.imshow('Background', bg)
      
    # Display
    cv2.imshow('Original', frame_gray)
    cv2.imshow("Mask", fgMask)

    if cv2.waitKey(10) == ord('q'): break

  else: break


  # Observations
  # - over time, the learning rate adapts and the background is hidden
  # - when lighting changes, the background is not filtered out and the LR has to re-adapt
  # - Intra frame: a static frame introduced in the sequence to cancel out "drifting" from previous frame differences. This explains the flickering every X frames. A video compressor minimizes this effect. Noise in the compression is another factor to consider (we loaded an MP4).