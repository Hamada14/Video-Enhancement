import cv2

class VideoReader:

    def __init__(self, source_video_path):
        self.video_capture = cv2.VideoCapture(source_video_path)

    def read_batch(self, batch_size):
        frame_number = 0;
        success, image = self.video_capture.read()
        frames = []
        while (frame_number < batch_size and success):
            frames.append(image)
            success, image = self.video_capture.read()
            frame_number += 1
        return frames
