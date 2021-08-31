import numpy as np
import argparse
import os
import time
import glob
import librosa
import sys
from datetime import datetime
import logging

from tensorflow.keras.models import load_model
from data_utils import generate_features, load_obj

def main(args):
    
    ids2intents = load_obj('ids2intents')
    ids2slots = load_obj('ids2slots')

    logging.info("\nIDs to Intents: {} \nIDs to Slots: {}".format(str(ids2intents.values()).replace("'", "\""), 
                                                                str(ids2slots.values()).replace("'", "\"")))


    audio_params = {
        "sampling_rate": args.sampling_rate,
        "min_freq": args.min_freq,
        "max_freq": args.max_freq,
        "win_size_ms": args.win_size_ms,
        "win_increase_ms": args.win_size_ms,
        "num_cepstral": args.num_cepstral
    }

    audio, sample_rate = librosa.load(args.wav_file, sr=16000, res_type='kaiser_best')
    audio = librosa.util.fix_length(audio, 16000*3)
    features = generate_features(True, audio, audio_params["sampling_rate"], 
                                          audio_params["win_size_ms"], audio_params["win_increase_ms"], 32, 
                                          audio_params['num_cepstral'], audio_params['min_freq'], audio_params['max_freq'])

    features = features['features']
    X = np.expand_dims(features, axis = 0)

    model = load_model(args.model_path)

    results = model(X, training=False)

    print(
f"""
Prediction

Raw indices: {np.argmax(results[0])} {np.argmax(results[1][0][0])} {np.argmax(results[1][0][1])}

Intent:{ids2intents[np.argmax(results[0])]} 
Slot1: {ids2slots[np.argmax(results[1][0][0])]}  Slot2: {ids2slots[np.argmax(results[1][0][1])]}\n
""")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Run inference on single .wav file')

    argparser.add_argument(
        '-w',
        '--wav_file',
        help='path to file')

    argparser.add_argument(
        '--model_path',
        required = False,
        help='path to model .h5 file')

    argparser.add_argument(
        '--sampling_rate',
        type=int,
        default=16000,
        help='Audio sampling rate')

    argparser.add_argument(
        '--min_freq',
        type=int,
        default=100,
        help='Spectrogram minimum frequency')

    argparser.add_argument(
        '--max_freq',
        type=int,
        default=8000,
        help='Spectrogram maximum frequency')        

    argparser.add_argument(
        '--win_size_ms',
        type=float,
        default=0.02,
        help='Spectrogram window size') 

    argparser.add_argument(
        '--num_cepstral',
        type=int,
        default=10,
        help='Number of MFCC cepstral coefficients') 

    args = argparser.parse_args()
    main(args)