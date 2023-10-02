from tensorflow.keras.utils import Sequence

import soundfile as sf
import librosa
import librosa.display
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, PitchShift, Shift, ClippingDistortion, Gain, LoudnessNormalization, TimeStretch
from tensorflow.python.ops import gen_audio_ops as contrib_audio

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import io, base64, os

DEBUG = False

class DatasetFactory:

    def __init__(self):
        self.actions = set()
        self.objects = set()
        self.locations = set()
        self.vocab = set()

    def get_query_slots(self, sentence):

        slots = [sentence[0].lower(), sentence[1].lower()]
        return slots

    def get_properties(self, data):

        data["action"] = data['action'].str.lower()
        data["object"] = data['object'].str.lower()
        data["location"] = data['location'].str.lower()

        actions = set(data.action.unique())
        objects = set(data.object.unique())
        locations = set(data.location.unique())

        return actions, objects, locations

    def get_vocab(self, actions, objects, locations, data):

        vocab = objects | locations

        if DEBUG:
            print(vocab)

        data["transcription"] = data['transcription'].str.replace('[^\w\s]','')
        data["transcription"] = data['transcription'].str.lower()

        for item in data.transcription:
            for word in item.split(" "):
                vocab.add(word)

        vocab = [s.strip() for s in vocab]

        return set(vocab)

    def add_corpora(self, data):

        actions, objects, locations = self.get_properties(data)
        vocab = self.get_vocab(actions, objects, locations, data)

        self.actions = set(self.actions | actions)
        self.objects = set(self.objects | objects)
        self.locations = set(self.locations | locations)
        self.vocab = set(self.vocab | vocab)
        self.query_slots = set(self.objects | self.locations)

    def get_slots_and_intents(self, intent_ids, slot_ids, data):

        slots = []
        for sentence in zip(data.object, data.location):
            slots.append(self.get_query_slots(sentence))

        vectorized_slots = list(map(lambda slots: np.array(list(map(lambda slot: slot_ids[slot], slots))), slots))
        vectorized_intents = list(map(lambda l: np.array([intent_ids[l]]), data.action))

        return vectorized_slots, vectorized_intents

    def process_data(self, data):

        self.actions = list(self.actions)
        self.objects = list(self.objects)
        self.locations = list(self.locations)
        self.vocab = list(self.vocab)
        self.query_slots = list(self.query_slots)

        word_ids, slot_ids, intent_ids = {' ': 0}, {}, {self.actions[i]: i for i in range(0, len(self.actions))}

        i = 0
        for slot in self.query_slots:
            if slot == 'none':
                continue
            slot_ids[slot] = i
            i += 1

        slot_ids['none'] = i

        #convert vocab to dictionary
        start = 1
        for i in range(len(self.vocab)):
            word_ids[self.vocab[i]] = start + i
        word_ids['unknown'] =  i + 1

        #create reverse dicts
        ids2words = dict((v, k) for k, v in word_ids.items())
        ids2slots = dict((v, k) for k, v in slot_ids.items())
        ids2intents = dict((v, k) for k, v in intent_ids.items())

        n_vocab = len(ids2words)

        n_classes = len(ids2intents)
        n_slots = len(ids2slots)

        vectorized_slots, vectorized_intents = self.get_slots_and_intents(intent_ids, slot_ids, data)

        filepaths = data['path'].to_numpy()

        return slot_ids, intent_ids, ids2intents, ids2slots, vectorized_slots, vectorized_intents, filepaths


def save_obj(obj, name):
    with open('data/pkl/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('data/pkl/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


### Sound processing function

def generate_features(draw_graphs, raw_data, sampling_freq,
                      frame_length, frame_stride, num_filters,
                      num_cepstral, low_frequency, high_frequency):
    graphs = []

    raw_data = np.expand_dims(raw_data, axis = -1)
    window_size = int(sampling_freq * frame_length)
    stride = int(sampling_freq * frame_stride)

    spectrogram = contrib_audio.audio_spectrogram(
        raw_data,
        window_size=window_size,
        stride=stride,
        magnitude_squared=True)

    mfcc = contrib_audio.mfcc(
        spectrogram,
        sampling_freq,
        dct_coefficient_count=num_cepstral,
        upper_frequency_limit=high_frequency,
        lower_frequency_limit=low_frequency)

    mfcc = np.squeeze(mfcc)

    if draw_graphs:
        mfcc_graph = np.swapaxes(mfcc, 0, 1)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfcc_graph, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title='MFCC')
        buf = io.BytesIO()

        plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)

        buf.seek(0)
        image = (base64.b64encode(buf.getvalue()).decode('ascii'))

        buf.close()

        graphs.append({
            'name': 'Cepstral Coefficients',
            'image': image,
            'imageMimeType': 'image/svg+xml',
            'type': 'image'
        })

    return {
        'features': mfcc,
        'graphs': graphs,
        'output_config': {
            'type': 'spectrogram',
            'shape': {
                'width': mfcc.shape[1],
                'height': mfcc.shape[0]
            }
        }
    }


def create_aug_pipeline():
    try:
        aug_pipeline = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
        AddBackgroundNoise(sounds_path="data/wavs/background_noise", p=0.3),
        ClippingDistortion(p=0.3),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.1),
        Gain(p=0.2),
        TimeStretch(p=0.05)
        ])
    except AssertionError:
        aug_pipeline = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
        ClippingDistortion(p=0.3),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.1),
        Gain(p=0.2),
        TimeStretch(p=0.05)
        ])
        print("\nWARNING: Cannot find background noise sample! Continuing without background noise augmentation.\n")
    return aug_pipeline

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, entries, num_list, audio_params, batch_size, shuffle=True, to_fit=True, augment = True, vis = False):

        self.entries = entries
        self.audio_params = audio_params
        self.batch_size = batch_size
        print(self.audio_params)
        self.n_intents, self.n_slots = num_list

        self.len = 2
        self.aug_pipeline = None
        if augment:
            self.aug_pipeline = create_aug_pipeline()
        self.vis = vis
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.entries[0]) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = [self.entries[0][k] for k in indexes]

        Y_intent = [self.entries[1][k] for k in indexes]
        Y_slot = [self.entries[2][k] for k in indexes]

        # Generate data
        X = self._generate_X(X_batch)

        if self.to_fit:
            y = self._generate_y(Y_intent, Y_slot)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.entries[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, batch_items):

        X = np.zeros(shape = (self.batch_size, 150, self.audio_params['num_cepstral'], 1))

        for i, batch_item in enumerate(batch_items):
            prefix = "data"
            wav_file = os.path.join(prefix, batch_item)
            audio, sample_rate = librosa.load(wav_file, sr=16000, res_type='kaiser_best')
            audio = librosa.util.fix_length(audio, 16000*3)

            if self.aug_pipeline:
                audio = self.aug_pipeline(audio, sample_rate)

            if DEBUG:
                new_filename = os.path.join('samples', os.path.basename(batch_item.split('.')[0]+'aug.wav'))
                print("Sample: ", new_filename)
                print("--------------")
                sf.write(new_filename, audio, sample_rate,  subtype='PCM_16')

            output = generate_features(self.vis, audio, self.audio_params["sampling_rate"],
                                          self.audio_params["win_size_ms"], self.audio_params["win_increase_ms"], 32,
                                          self.audio_params['num_cepstral'], self.audio_params['min_freq'], self.audio_params['max_freq'])

            features = output['features']
            X[i, ] = np.expand_dims(features, axis = -1)
        return X

    def _generate_y(self, intents, slots):
        intent_y = np.empty((self.batch_size, self.n_intents), dtype=int)
        slot_y = np.empty((self.batch_size, self.len, self.n_slots), dtype=int)

        # Generate data
        for i, batch_item in enumerate(intents):
            intent = intents[i]
            slot = slots[i]
            intent_y[i,] = np.eye(self.n_intents)[intent]
            slot_y[i,] = np.eye(self.n_slots)[slot][np.newaxis, :]

        return [intent_y, slot_y]

