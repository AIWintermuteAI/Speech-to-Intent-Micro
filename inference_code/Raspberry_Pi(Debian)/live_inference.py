import pyaudio
import sys
import time
#import librosa
import numpy as np
import wave
import pickle
import threading
from array import array
from queue import Queue, Full
import argparse

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio

SEC = 3
CHUNK_SIZE = 128
RATE = 16000
FRAME_LENGTH = 0.02
FRAME_STRIDE = 0.02
NUM_CEPSTRAL=10
MIN_FREQ = 100
MAX_FREQ = RATE//2
NUM_CHUNKS = RATE*SEC/CHUNK_SIZE
BUF_MAX_SIZE = CHUNK_SIZE * 10

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class InferenceEngine:

  def __init__(self, model_file):
    self.ids2intents = load_obj('ids2intents')
    self.ids2slots = load_obj('ids2slots')

    self.interpreter = tf.lite.Interpreter(model_file)
    self.interpreter.allocate_tensors()

    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
    self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]
    self.input_index = self.input_details[0]['index']

    _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']

  def infer(self, data):
    start_time = time.time()
    if self.input_details[0]['dtype'] == np.int8:
        data = np.asarray(data/self.input_scale + self.input_zero_point, dtype=np.int8)
    self.interpreter.set_tensor(self.input_index, data)
    self.interpreter.invoke()
    #output_details = self.interpreter.get_output_details()[0]
    #output = np.squeeze(self.interpreter.get_tensor(output_details['index']))

    output_data_slot = np.asarray(self.interpreter.get_tensor(self.output_details[0]['index']), dtype=np.float32)
    output_data_intent = np.asarray(self.interpreter.get_tensor(self.output_details[1]['index']), dtype=np.float32)

    # If the model is quantized (uint8 data), then dequantize the results
    if self.output_details[0]['dtype'] == np.int8:
        output_data_intent = (output_data_intent - self.output_zero_point) * self.output_scale
        output_data_slot = (output_data_slot - self.output_zero_point) * self.output_scale

    intent = self.ids2intents[np.argmax(output_data_intent[0])]
    slots = []
    slot_argmax = np.argmax(output_data_slot[0], axis=1)
    for i in range(len(slot_argmax)):
        slots.append(self.ids2slots[slot_argmax[i]])

    output = (intent, slots)
    elapsed_ms = (time.time() - start_time) * 1000
    print("Inference time: ", elapsed_ms)
    #print(output)
    return output


class Recorder:

    def __init__(self, source, threshold, engine):
        self.recording = False
        self.min_volume = threshold
        self.engine = engine
        self.source = source
        self.stoped = False

        q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

        if self.source:
            listen_t = threading.Thread(target=self.play, args=(q,))
        else:
            listen_t = threading.Thread(target=self.listen, args=(q,))
        listen_t.start()

        record_t = threading.Thread(target=self.record, args=(q,))
        record_t.start()

        try:
            while True:
                listen_t.join(0.1)
                record_t.join(0.1)
        except KeyboardInterrupt:
            self.stoped = True

        listen_t.join()
        record_t.join()


    def predict(self, data):

        data = np.array(data).astype(np.float32)
        data = data.flatten()/32768
        #mfcc = librosa.feature.mfcc(data, 16000, n_mfcc=40)
        data = np.expand_dims(data, axis = -1)
        window_size = int(RATE * FRAME_LENGTH)
        stride = int(RATE * FRAME_STRIDE)
        
        spectrogram = contrib_audio.audio_spectrogram(
            data,
            window_size=window_size,
            stride=stride,
            magnitude_squared=True)
        
        mfcc = contrib_audio.mfcc(
            spectrogram,
            RATE,
            dct_coefficient_count=NUM_CEPSTRAL,
            upper_frequency_limit=MAX_FREQ, 
            lower_frequency_limit=MIN_FREQ)
        
        #mfcc = np.squeeze(mfcc)

        x = np.reshape(mfcc, [-1, 150, 10, 1])
        predictions = self.engine.infer(x)
        parsed_predictions = f"""
        --------------++++------------------
        Prediction
        Intent:{predictions[0]} 
        Slot1: {predictions[1][0]}  Slot2: {predictions[1][1]}\n
        --------------++++------------------
        """
        print(parsed_predictions)
        return predictions

    def record(self, q):

        frame_buffer = []
        while True:
            if self.stoped:
                break
            chunk = q.get()
            vol = max(chunk)
            frame_buffer.append(chunk)

            if self.recording:
                frames.append(chunk)
                if len(frames) >= NUM_CHUNKS:
                   result = self.predict(frames)
                   #if result > 0.3: print("Triggered")
                   self.recording = False
            else:
                pass
                
            if vol >= self.min_volume and not self.recording:
                self.recording = True
                frames = frame_buffer[-7:]
     

    def play(self, q):
        wf = wave.open(self.source, 'rb')
        data = wf.readframes(CHUNK_SIZE)
        while data != b'':
            try:
                q.put(array('h', data))
            except Full:
                pass  # discard
            data = wf.readframes(CHUNK_SIZE)
        self.stoped = True

    def listen(self, q):
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        while True:
            if self.stoped:
                break
            try:
                q.put(array('h', stream.read(CHUNK_SIZE)))
            except Full:
                pass  # discard

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Record sound samples')

    argparser.add_argument(
        '-s',
        '--source',
        default=None,
        help='Source:mic or file')

    argparser.add_argument(
        '-t',
        '--threshold',
        default=900,
        help='Silence threshold')
        
    argparser.add_argument(
        '-m',
        '--model',
        default='good_model.tflite',
        help='Path to model file')
        
    args = argparser.parse_args()

    engine = InferenceEngine(args.model)
    recorder = Recorder(args.source, args.threshold, engine)