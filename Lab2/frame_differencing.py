import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("..\Lab1\material\Video.mp4")

assert video.isOpened(), "Cannot load video"

N = 2
THRESHOLD = 60
MAX_VAL = 255
frames = []
motion = []

# Read frame by frame
while(video.isOpened()):
  
  ret, frame = video.read()

  if ret == True:

    # Grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(frame_gray)

    # Motion mask
    if len(frames) >= N+1:
        diff = cv2.absdiff(frames[-1], frames[-(1+N)])
        _, motion_mask = cv2.threshold(diff, THRESHOLD, MAX_VAL, cv2.THRESH_BINARY)
        motion.append(motion_mask)

        cv2.imshow("Motion", motion_mask)
        cv2.imshow("Frame", frame_gray)
        if cv2.waitKey(2) == ord('q'): break

  else: break


  # Observations
  # - for high N, objects are multiplied (shapes are too distant and don't cancel each other)
