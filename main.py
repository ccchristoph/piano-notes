import pyaudio
import numpy as np
import time
import random
import threading

# Sample rate and chunk size
SAMPLE_RATE = 44100

# Mapping of piano keys and frequencies
piano_freq_full = {
    'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 'G1': 49.00, 'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.26, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
    'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02, 'E7': 2637.02, 'F7': 2793.83, 'F#7': 2959.96, 'G7': 3135.96, 'G#7': 3322.44, 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07,
    'C8': 4186.01
}

piano_freq = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.26, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
}

class AudioRecorder:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk = 4096*4
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk)
        self.latest_audio_data = np.array([])
        self.is_recording = True
        self.read_audio_thread = threading.Thread(target=self.record_audio)
        self.read_audio_thread.start()
    
    def record_audio(self):
        while self.is_recording:
            audio_data = np.frombuffer(self.stream.read(self.chunk), dtype=np.float32)
            self.latest_audio_data = audio_data


    def get_audio_data(self):
        return self.latest_audio_data

    def stop(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def analyze_audio(data):

    threshold = 5
    if data.size == 0:
        print("Empty audio data received.")
        return 0
    fourier = np.fft.fft(data)
    magnitudes = np.abs(fourier)

    freqs = np.fft.fftfreq(len(fourier))
    
    idx = np.argmax(magnitudes)
    max_magnitude = magnitudes[idx]
    
    detected_frequency = abs(freqs[idx] * 44100)
    if max_magnitude < threshold:
        detected_frequency = 0
    #print(f"Frequency: {detected_frequency:.3f} \t Fourier Magnitude: {max_magnitude:.3f}")
    return detected_frequency

def get_adjacent_notes(target_note, piano_freq):
    notes = list(piano_freq.keys())
    idx = notes.index(target_note)
    lower_note = piano_freq[notes[idx - 1]] if idx > 0 else None
    upper_note = piano_freq[notes[idx + 1]] if idx < len(notes) - 1 else None
    return lower_note, upper_note

def check_frequency(target_frequency, detected_frequency, target_note):
    lower_note, upper_note = get_adjacent_notes(target_note, piano_freq)
    if lower_note:
        lower_limit = (target_frequency + lower_note) / 2
    else:
        lower_limit = target_frequency * 0.9  # 10% lower for edge case
    if upper_note:
        upper_limit = (target_frequency + upper_note) / 2
    else:
        upper_limit = target_frequency * 1.1  # 10% higher for edge case
    return lower_limit <= detected_frequency <= upper_limit

def main():

    audio_recorder = AudioRecorder()
    while(audio_recorder.get_audio_data().size == 0):
        time.sleep(0.1)

    start_time = time.time()

    while True:
        target_note = random.choice(list(piano_freq.keys()))
        print(f"Note to play: {target_note}")
        target_frequency = piano_freq[target_note]
        detected_frequency = 0
        
        start_time_note = time.time()
        while(not check_frequency(target_frequency, detected_frequency, target_note)):
            audio_data = audio_recorder.get_audio_data()
            detected_frequency = analyze_audio(audio_data)
            #print(detected_frequency)
            time.sleep(0.05)  # Sleep for 50ms

        end_time_note = time.time()
        print(f"Correct! Time taken for {target_note}: {end_time_note-start_time_note}")

if __name__ == "__main__":
    main()
