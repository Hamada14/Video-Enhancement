import logging

import cv2

logger = logging.getLogger()

class VideoReader:

    def __init__(self, source_video_path):
        self.source_video_path = source_video_path
        self.video_capture = cv2.VideoCapture(source_video_path)

    def read_batch(self, batch_size):
        success, image = self.video_capture.read()
        if not success:
            logger.info('Could not find the video in the specified location {' + self.source_video_path + '}')
        frames = []
        frame_number = 0
        while (frame_number < batch_size and success):
            frames.append(image)
            success, image = self.video_capture.read()
            frame_number += 1
        return frames
