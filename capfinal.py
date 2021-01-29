#!/usr/bin/env python
# coding: utf-8

# In[25]:


import sys
import cv2
from random import randint
from tqdm import tqdm
import time


# In[26]:


def create_tracker_by_name(tracker_type="MIL"):
    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None

    return tracker


# In[27]:


tracker = create_tracker_by_name()


# In[28]:


file = r"C:\Users\Swraj Palekar\Downloads\Lionel Messi 2nd Goal Vs Chelsea Home UCl 15_03_2018"
output_filepath = "out_" + file + ".avi"
video = cv2.VideoCapture(file+".mp4")


# In[29]:


cap = cv2.VideoCapture(r"C:\Users\Swraj Palekar\Downloads\Lionel Messi 2nd Goal Vs Chelsea Home UCl 15_03_2018.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )




# In[35]:


success, frame = video.read()
if not success:
    print("Cannot read video")
video_size = (frame.shape[0], frame.shape[2])    


# In[37]:


boxes = []
colors = []
while True:
    k = cv2.waitKey(0) & 0xFF
    if (k == 113): # q is pressed
        break
    box = cv2.selectROI("MultiTracker", frame)
    boxes.append(box)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
cv2.destroyAllWindows()    
print("bounding boxes are {}".format(boxes))


# In[7]:


multi_tracker = cv2.MultiTracker_create()
tracking_boxes = []
for box in boxes:
    multi_tracker.add(create_tracker_by_name("CSRT"), frame, box)
output_frames = []
start_track_time = time.time()


# In[8]:


while video.isOpened():
    success, frame = video.read()
    if not success:
        break


# In[9]:


success, boxes = multi_tracker.update(frame)


# In[10]:


if success:
        for i, new_box in enumerate(boxes):
            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        tracking_boxes.append(boxes)
     
        cv2.imshow("Multi Tracker", frame)
        out_frames.append(frame)
       
                
       


# In[80]:


tracking_time = time.time() - start_track_time
print("Tracking time: {} [s]".format(tracking_time))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
frame_per_second = 24
is_color = 1
writer = cv2.VideoWriter(output_filepath, fourcc, frame_per_second, video_size, is_color)
pbar = tqdm(range(len(output_frames)), unit="frame")


# In[81]:


start_writing_time = time.time()
for i_frame in pbar:
    frame = out_frames[i_frame]
    writer.write(frame)
writing_time = time.time() - start_writing_time
print("Writing time: {} [s]".format(writing_time))
writer.release()
cv2.destroyAllWindows()     


# In[82]:


while True:
    ret, frame= video.read()
    if not ret:
        break
        
    (success, boxes) = multi_tracker.update(frame)
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,205,200), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    video.release()

    


# In[ ]:





# In[ ]:




