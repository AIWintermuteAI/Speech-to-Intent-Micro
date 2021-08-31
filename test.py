import pandas as pd
import numpy as np
import argparse
import os
import time
import glob
import sys
from datetime import datetime
import logging

from tensorflow.keras.models import load_model

from data_utils import save_obj, load_obj, DatasetFactory, DataGenerator

def test_models(test_generator, accuracy_threshold, model_name = None, model_directory = None):
    
    print("Testing")
    if model_directory:
        model_files_list = []
        file_search = lambda ext : glob.glob(model_directory + ext, recursive=True)
        for ext in ['/**/*.h5']: model_files_list.extend(file_search(ext))
    else:
        model_files_list = [model_name]

    for model_file in model_files_list:

        model = load_model(model_file)

        intent_correct = 0
        slot_correct = 0

        for num in range(len(test_generator)):

            X, y = test_generator.__getitem__(num)

            try:
                results = model(X, training=False)
            except Exception as e:
                print('Error')
                break

            if np.argmax(y[0]) == np.argmax(results[0]):
                intent_correct += 1

            if np.argmax(y[1][0][0]) == np.argmax(results[1][0][0]):
                slot_correct += 1 

            if np.argmax(y[1][0][1]) == np.argmax(results[1][0][1]):
                slot_correct += 1     

        accuracy_intent = intent_correct/len(test_generator)
        accuracy_slot = slot_correct/(len(test_generator)*2)

        if (accuracy_intent < accuracy_threshold or accuracy_slot < accuracy_threshold) and model_directory:
            continue
            
        model.summary()    
        
        print(
f"""Model {model_file}

Accuracy Intent {accuracy_intent} %
Accuracy Slot {accuracy_slot} %

""")


def main(args):

    assert (args.model_path or args.model_folder_path), "model_path or model_folder_path argument should be specified"

    test_data = pd.read_csv(args.test_dataset_path)

    test_data["action"] = test_data['action'].str.lower()
    test_data["object"] = test_data['object'].str.lower()
    test_data["location"] = test_data['location'].str.lower()

    dataset_processor = DatasetFactory()

    filepaths_test = test_data['path'].to_numpy()
    
    ids2intents = load_obj('ids2intents')
    ids2slots = load_obj('ids2slots')

    slot_ids = load_obj('slot_ids')
    intent_ids = load_obj('intent_ids')

    logging.info("\nIDs to Intents: {} \nIDs to Slots: {}".format(str(ids2intents.values()).replace("'", "\""), 
                                                                str(ids2slots.values()).replace("'", "\"")))

    vectorized_slots_test, vectorized_intents_test = dataset_processor.get_slots_and_intents(intent_ids, slot_ids, test_data)

    save_obj(vectorized_slots_test, 'vectorized_slots_test')
    save_obj(vectorized_intents_test, 'vectorized_intents_test') 

    n_classes = len(ids2intents)
    n_slots = len(ids2slots)

    audio_params = {
        "sampling_rate": args.sampling_rate,
        "min_freq": args.min_freq,
        "max_freq": args.max_freq,
        "win_size_ms": args.win_size_ms,
        "win_increase_ms": args.win_size_ms,
        "num_cepstral": args.num_cepstral
    }

    test_generator = DataGenerator([filepaths_test, vectorized_intents_test, vectorized_slots_test], 
                                        [n_classes,n_slots], audio_params, batch_size = 1,
                                        shuffle=False, to_fit=True, augment = False)

    data = test_generator.__getitem__(0)

    logging.debug("""Test input data shape: {}
    Test Intent Output Shape: {}
    Test Slot Output Shape: {}
    Test of batches : {}""".format(data[0].shape,
                                    data[1][0].shape,
                                    data[1][1].shape,
                                    test_generator.__len__()))

    test_models(test_generator, args.min_accuracy, model_name = args.model_path, model_directory = args.model_folder_path)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Test Speech to Intent model on FLUENT Speech Commands-like dataset')

    argparser.add_argument(
        '-v',
        '--test_dataset_path',
        default="data/csv/test_data.csv",
        help='path to validation data .csv file')

    argparser.add_argument(
        '--model_path',
        required = False,
        help='path to model .h5 file')

    argparser.add_argument(
        '--model_folder_path',
        required = False,
        help='path to directory with .h5 model files')

    argparser.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='Batch size for training and validation')

    argparser.add_argument(
        '--min_accuracy',
        default=0.7,
        type=float,
        help='Minimum accuracy to display test results')

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