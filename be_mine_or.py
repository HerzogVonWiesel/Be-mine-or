from ultralytics import YOLO
import cv2

import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import threading
import time


class AudioPlayer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_segment = AudioSegment.from_file(file_path)
        self.volume = 1.0  # Default volume (1.0 is 100%)
        self.chunk_size = 1024  # 1KB chunks for playback
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.stop_event = threading.Event()
        self.play_thread = threading.Thread(target=self._play_loop)
        self.play_thread.start()

    def _play_loop(self):
        audio_data = np.array(self.audio_segment.get_array_of_samples())
        sample_width = self.audio_segment.sample_width
        channels = self.audio_segment.channels
        rate = self.audio_segment.frame_rate
        audio_format = self.p.get_format_from_width(sample_width)
        num_samples = len(audio_data)

        if self.stream is None:
            self.stream = self.p.open(format=audio_format,
                                      channels=channels,
                                      rate=rate,
                                      output=True)

        idx = 0
        while not self.stop_event.is_set():
            end_idx = min(idx + self.chunk_size, num_samples)
            chunk = audio_data[idx:end_idx]
            chunk = (chunk * self.volume).astype(np.int16)
            self.stream.write(chunk.tobytes())
            idx = end_idx if end_idx < num_samples else 0

    def set_volume(self, volume):
        self.volume = volume

    def stop(self):
        self.stop_event.set()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.play_thread.join()


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 256)

# load custom model in current directory
model = YOLO('be-mine-or-clsm_openvino_model/') # YOLO('be-mine-or-clsm_openvino_model/') # YOLO('be-mine-or-clsm.pt')

# object classes
classNames = ["NONE", "holding-hands", "hugging", "kissing"]
volumes = [1.0, 0.6, 0.3, 0.0]

noisePlayer = AudioPlayer("city_noise.mp3")

# use webcam to classify image
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions
    results = model(frame, verbose=True, stream=True)

    # Process classification results, not detection
    for result in results:
        cls = result.probs.top1
        conf = result.probs.top1conf
        if conf > 0.5:  # Confidence threshold
            # Adjust the volume based on confidence
            noiseVolume = volumes[cls]
            noisePlayer.set_volume(noiseVolume)
            label = f'{classNames[cls]}: {conf:.2f}'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(noiseVolume), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)
    time.sleep(0.5)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()