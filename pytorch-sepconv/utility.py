import cv2
import os
from matplotlib import pyplot as plt

def get_full_frame_path(frames_folder, frame_format):
    return frames_folder + os.sep + frame_format

def extract_frames(video_path, frames_folder, frame_format):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    index = 0
    success = True
    destination_path = get_full_frame_path(frames_folder, frame_format)
    while success:
        # print(index)
        success,image = vidcap.read()
        cv2.imwrite(destination_path % (index * 2), image)
        if cv2.waitKey(10) == 27:
            break
        index += 1
    return index


def build_video(frames_folder, frame_format, fps, number_of_frames, video_path, extension):
    if(number_of_frames <= 0):
        return
    frames_path = get_full_frame_path(frames_folder, frame_format)
    frame = cv2.imread(frames_path % (0))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width,height))
    for i in range(number_of_frames + 1):
        video.write(cv2.imread(frames_path % (i)))
    cv2.destroyAllWindows()
    video.release()
