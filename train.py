import pandas as pd
import numpy as np
import argparse
import os
import time
import sys
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from data_utils import save_obj, load_obj, DatasetFactory, DataGenerator
from models import get_model, tflite_convert, tflite_micro_convert

import logging

def main(args):
    batch_size = args.batch_size
    generate_data = args.generate_data

    train_data = pd.read_csv(args.train_dataset_path)
    valid_data = pd.read_csv(args.valid_dataset_path)

    audio_params = {
        "sampling_rate": args.sampling_rate,
        "min_freq": args.min_freq,
        "max_freq": args.max_freq,
        "win_size_ms": args.win_size_ms,
        "win_increase_ms": args.win_size_ms,
        "num_cepstral": args.num_cepstral
    }

    if generate_data:

        dataset_processor = DatasetFactory()

        dataset_processor.add_corpora(train_data)
        dataset_processor.add_corpora(valid_data)

        slot_ids, intent_ids, ids2intents, ids2slots, vectorized_slots_train, vectorized_intents_train, filepaths_train = dataset_processor.process_data(train_data)
        _slot_ids, _intent_ids, _ids2intents, _ids2slots, vectorized_slots_valid, vectorized_intents_valid, filepaths_valid = dataset_processor.process_data(valid_data)

        assert ids2intents == _ids2intents
        assert ids2slots == _ids2slots

        save_obj(ids2intents, 'ids2intents')
        save_obj(ids2slots, 'ids2slots')

        save_obj(slot_ids, 'slot_ids')
        save_obj(intent_ids, 'intent_ids')

        save_obj(vectorized_slots_train, 'vectorized_slots_train')
        save_obj(vectorized_intents_train, 'vectorized_intents_train')

        save_obj(vectorized_slots_valid, 'vectorized_slots_valid')
        save_obj(vectorized_intents_valid, 'vectorized_intents_valid')

    else:

        dataset_processor = DatasetFactory()

        filepaths_train = train_data['path'].to_numpy()
        filepaths_valid = valid_data['path'].to_numpy()

        ids2intents = load_obj('ids2intents')
        ids2slots = load_obj('ids2slots')

        slot_ids = load_obj('slot_ids')
        intent_ids = load_obj('intent_ids')

        vectorized_slots_train, vectorized_intents_train = dataset_processor.get_slots_and_intents(intent_ids, slot_ids, train_data)
        vectorized_slots_valid, vectorized_intents_valid = dataset_processor.get_slots_and_intents(intent_ids, slot_ids, valid_data)

        #vectorized_slots_train = load_obj('vectorized_slots_train')
        #vectorized_intents_train = load_obj('vectorized_intents_train')

        #vectorized_slots_valid = load_obj('vectorized_slots_valid')
        #vectorized_intents_valid = load_obj('vectorized_intents_valid')

    logging.info("\nIDs to Intents: {} \nIDs to Slots: {}".format(str(ids2intents.values()).replace("'", "\""),
                                                                str(ids2slots.values()).replace("'", "\"")))

    n_classes = len(ids2intents)
    n_slots = len(ids2slots)

    training_generator = DataGenerator([filepaths_train, vectorized_intents_train, vectorized_slots_train],
                                    [n_classes, n_slots], audio_params, batch_size = batch_size,
                                    shuffle=True, to_fit=True, augment = True)

    _data = training_generator.__getitem__(0)

    logging.debug("""Train input data shape: {}
    Train Intent Output Shape: {}
    Train Slot Output Shape: {}
    Number of batches : {}""".format(_data[0].shape,
                                    _data[1][0].shape,
                                    _data[1][1].shape,
                                    training_generator.__len__()))


    validation_generator = DataGenerator([filepaths_valid, vectorized_intents_valid, vectorized_slots_valid],
                                        [n_classes,n_slots], audio_params, batch_size = batch_size,
                                        shuffle=False, to_fit=True, augment = False)

    _data = validation_generator.__getitem__(0)

    logging.debug("""Validation input data shape: {}
    Validation Intent Output Shape: {}
    Validation Slot Output Shape: {}
    Validation of batches : {}""".format(_data[0].shape,
                                    _data[1][0].shape,
                                    _data[1][1].shape,
                                    validation_generator.__len__()))

    model = get_model(args.model_type, n_classes, n_slots, audio_params)

    optim = Adam(learning_rate=args.lr)

    model.compile(optimizer = optim, loss='categorical_crossentropy', metrics='accuracy')
    model.summary()

    output_path = os.path.join("checkpoints", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_path)
    print("Project folder: {}".format(output_path))

    model_name = os.path.join(output_path, "slu_model.h5")

    my_callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, verbose = 1),
        ModelCheckpoint(filepath=model_name, save_best_only=True, verbose = 1),
        TensorBoard(log_dir='./logs'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose = 1)]
    try:
        model.fit(training_generator, validation_data = validation_generator,
                callbacks = my_callbacks, epochs=args.epochs,
                workers = 4, max_queue_size = 10,
                use_multiprocessing = False)
    except KeyboardInterrupt:
        raise

    calibration_generator = DataGenerator([filepaths_valid, vectorized_intents_valid, vectorized_slots_valid],
                                        [n_classes,n_slots], audio_params, batch_size = 1,
                                        shuffle=False, to_fit=True, augment = False)

    tflite_filename = tflite_convert(model, model_name, calibration_generator)
    tflite_micro_convert(tflite_filename)

if __name__ == "__main__":


    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Train and validate Speech to Intent model on FLUENT Speech Commands-like dataset')

    argparser.add_argument(
        '-t',
        '--train_dataset_path',
        default="data/csv/train_data.csv",
        help='path to train data .csv file')

    argparser.add_argument(
        '-v',
        '--valid_dataset_path',
        default="data/csv/valid_data.csv",
        help='path to validation data .csv file')

    argparser.add_argument(
        '-m',
        '--model_type',
        default="plain_Conv2D",
        help='type of model to train: plain_Conv2D, DW_Conv2D, res_Conv2D')

    argparser.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='Batch size for training and validation')

    argparser.add_argument(
        '-l',
        '--lr',
        default=1e-3,
        type=float,
        help='Initial learning rate')

    argparser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train')

    argparser.add_argument(
        '-d',
        '--generate_data',
        default=False,
        help='Whether or not to re-generate intents and slots data')

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
    print(args)
    main(args)
