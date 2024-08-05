from ultralytics import YOLO
import cv2

import numpy as np
import pyaudio
from pydub import AudioSegment
import threading
import time

class AudioPlayer:
    def __init__(self, file_path, calming=False):
        self.file_path = file_path
        self.calming = calming
        self.audio_segment = AudioSegment.from_file(file_path)
        self.volume = 1.0  # Default volume (1.0 is 100%)
        self.future_volume = 1.0
        self.chunk_size = 1024  # 1KB chunks for playback
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.stop_event = threading.Event()
        self.play_thread = threading.Thread(target=self._play_loop)
        self.play_thread.start()
        self.volume_thread = threading.Thread(target=self._gradual_volume)
        self.volume_thread.start()

    def _gradual_volume(self):
        calmingFactor = 0.5 if self.calming else 1.0
        stressFactor = 1.0 if self.calming else 0.5
        while not self.stop_event.is_set():
            if self.volume < self.future_volume:
                self.volume = min(self.volume + 0.05*stressFactor, self.future_volume)
            elif self.volume > self.future_volume:
                self.volume = max(self.volume - 0.05*calmingFactor, self.future_volume)
            time.sleep(0.5)

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
        self.future_volume = volume

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

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 256)

# load custom model in current directory
model = YOLO('be-mine-or-clsm.pt') # YOLO('be-mine-or-clsm_openvino_model/') # YOLO('be-mine-or-clsm.pt')

# object classes
classNames = ["no affection", "holding hands", "hugging", "kissing"]
noiseVolumes = [1.0, 0.6, 0.15, 0.0]
zenVolumes = [0.0, 0.0, 0.6, 1.0]

noisePlayer = AudioPlayer("audio/city_noise.mp3")
zenPlayer = AudioPlayer("audio/zen.mp3")

# use webcam to classify image
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions
    results = model(frame, verbose=False, stream=True)

    # Process classification results, not detection
    for result in results:
        cls = result.probs.top1
        conf = result.probs.top1conf
        if conf > 0.5:  # Confidence threshold
            # Adjust the volume based on confidence
            noiseVolume = noiseVolumes[cls]
            zenVolume = zenVolumes[cls]
            noisePlayer.set_volume(noiseVolume)
            zenPlayer.set_volume(zenVolume)
            label = f'{classNames[cls]}: {conf:.2f}'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "NoiseVol:"+str(noisePlayer.volume), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
            cv2.putText(frame, "CalmVol:"+str(zenPlayer.volume), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 90, 90), 2)

    cv2.imshow('Webcam', frame)
    time.sleep(0.9)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()