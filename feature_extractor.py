import warnings

import numpy as np

def vectorized_stride_v2(data, sub_window_size,
                         stride_size, time_index=0, max_time=None):

# Function to window audio data

    if max_time is None:
        # max_time = len(data.flatten()) - sub_window_size + 1
        max_time = data.shape[0] - sub_window_size + 1

    # start = time_index + 1 - sub_window_size + 1
    start = time_index

    sub_windows = (
            start +
            np.expand_dims(np.arange(sub_window_size), 0) +
            # Create a rightmost vector as [0, V, 2V, ...].
            # np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
            np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )

    start_windows = sub_windows[:, 0]

    return data[sub_windows], start_windows


class FeatureExtractor:
    def __init__(self, data, fs=16000, n_chunks=2, n_mfcc=40, Lw=512, win_len=512, n_fft=512, hop_len=128, eth=None, channel_data=None, ref=None):
        from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_flatness, mfcc, zero_crossing_rate, \
            spectral_rolloff, spectral_contrast, tonnetz, delta

        # Input of the class feature extractor is windowed audio data. 
        # NB now each function returns also the name of the feature(s) being extracted. Do adaptations within codes using them
        # The class extract audio features from defined fuctions (e.g. spectral_centroid from librosa) and calculates statisticas, e.g. mean/variance of the feature for the provided windows.
        self.data = data
        self.fs = fs
        self.center = False  # do not center frame when padding for spectral features
        self.n_mfcc = n_mfcc
        self.win_len = win_len
        self.n_fft = n_fft
        self.ref = ref
        self.hop_len = hop_len  # 75% overlap
        self.n_chunks = n_chunks
        self.Lw = Lw
        self.eth = eth
        self.matrix = channel_data
        self.window = 'hann'
        self.smoothing_level_perc = 10
        self.threshold_level = 10
        self.high_pass = False
        self.order = 16  # for LPC features
        self.npncc = 13

        assert len(self.data) > 0, "Length of y is 0!"

        self.args = {'y': self.data, 'sig': self.data, 'ref': self.ref, 'sr': self.fs, 'fs': self.fs, 'n_fft': self.n_fft, 'n_mfcc': self.n_mfcc, 'win_length': self.win_len,
                     'hop_length': self.hop_len, 'n_chunks': self.n_chunks, 'Lw': self.Lw, 'eth': self.eth, 'center': self.center,
                     'frame_length': self.win_len, 'matrix': self.matrix, 'window': self.window, 'smoothing_level_perc': self.smoothing_level_perc,
                     'threshold_level': self.threshold_level, 'high_pass': self.high_pass, 'order': self.order,
                     'num_ceps': self.npncc, 'win_len': self.win_len/self.fs, 'win_hop': self.hop_len/self.fs, 'win_type': self.window, 'nfft': self.n_fft}

        self.features = {}

    def feature_full(self, function, name):
        # does not calculate any statistics, just returns the features for given windows.
        self.features[f'{name}'] = function

    def feature_delta_full(self, function, params) -> object:
        # 1st derivative
        self.features[f'{function.__name__}_delta'] = np.diff(function(**params), prepend=function(**params)[:, 0, np.newaxis], axis=1)

    def feature_mean(self, function, name, chunk_id=1, tot_chunks=1):
        if function.shape[0] == 1 or len(function.shape) < 2:
            self.features['{}_mean_chunk{}of{}'.format(name, chunk_id, tot_chunks)] = [np.mean(function)]
        else:
            for row in range(function.shape[0]):
                self.features['{}{}_mean_chunk{}of{}'.format(name, row+1, chunk_id, tot_chunks)] = [np.mean(function[row, :])]

    def feature_variance(self, function, name, chunk_id=1, tot_chunks=1):
        if function.shape[0] == 1 or len(function.shape) < 2:
            self.features['{}_variance_chunk{}of{}'.format(name, chunk_id, tot_chunks)] = [np.var(function)]
        else:
            for row in range(function.shape[0]):
                self.features['{}{}_variance_chunk{}of{}'.format(name, row+1, chunk_id, tot_chunks)] = [np.var(function[row, :])]

    def feature_delta(self, function, params):
        # this function only works for spectral features from librosa
        res = function(**params)
        if res.size == 1:
            raise ValueError('Cannot calculate delta of a feature vector of length 1!')
        elif res.shape[0] == 1 and res.size > 1:
            deltas = np.zeros((res.size-1,))
            for row in range(res.size-1):
                deltas[row] = res[0, row+1] - res[0, row]
            self.features['{}_delta1of1'.format(function.__name__)] = [deltas.mean()]
        elif res.shape[0] > 1 and res.size > 1:
            deltas = np.zeros((res.shape[0], res.shape[1]-1))
            for row in range(res.shape[0]):
                for col in range(res.shape[1]-1):
                    deltas[row, col] = res[row, col+1] - res[row, col]
                self.features['{}{}_delta1of1'.format(function.__name__, row+1)] = [deltas[row, :].mean()]

    def feature_delta2(self, function, params):
        # 2nd derivative
        # this function only works for spectral features from librosa
        res = function(**params)
        if res.size == 1:
            raise ValueError('Cannot calculate delta of a feature vector of length 1!')
        elif res.shape[0] == 1 and res.size > 1:
            deltas2 = np.zeros((res.size-2,))
            for row in range(res.size - 2):
                deltas2[row] = \
                    (res[0, row + 2] - res[0, row+1]) - (res[0, row + 1] - res[0, row])
            self.features['{}_delta21of1'.format(function.__name__)] = [deltas2.mean()]
        elif res.shape[0] > 1 and res.size > 1:
            deltas2 = np.zeros((res.shape[0], res.shape[1]-2))
            for row in range(res.shape[0]):
                for col in range(res.shape[1] - 2):
                    deltas2[row, col] = (res[row, col + 2] - res[row, col+1]) - (res[row, col + 1] - res[row, col])
                self.features['{}{}_delta21of1'.format(function.__name__, row + 1)] = [deltas2[row, :].mean()]

    def run_function(self, function, stats='mean'):
        import inspect
        required_args = inspect.getfullargspec(function).args
        parameters = {}
        parameters['y'] = self.args['y']
        for arg in required_args:
            if arg in self.args.keys():
                parameters[arg] = self.args[arg]
        if function.__name__ == 'mfcc':
            parameters['hop_length'] = self.args['hop_length']
            parameters['n_fft'] = self.args['n_fft']
            parameters['win_length'] = self.args['win_length']
            parameters['center'] = self.args['center']
            parameters['n_mfcc'] = self.args['n_mfcc']

        if stats == 'full':
            self.feature_full(function(**parameters), function.__name__)
        elif stats == 'mean':
            self.feature_mean(function(**parameters), function.__name__)
        elif stats == 'mean_chunks':
            chunks = np.array_split(self.data, self.n_chunks)
            for i, chunk in enumerate(chunks):
                parameters['y'] = chunk
                self.feature_mean(function(**parameters), function.__name__, chunk_id=i+1, tot_chunks=self.n_chunks)
        elif stats == 'variance':
            self.feature_variance(function(**parameters), function.__name__)
        elif stats == 'delta':
            self.feature_delta(function, parameters)
        elif stats == 'delta_full':
            self.feature_delta_full(function, parameters)
        elif stats == 'delta2':
            self.feature_delta2(function, parameters)
        elif stats == 'one_chunk':
            warnings.warn("one_chunk feature should be updated before being used.")
            if 'win_length' in parameters.keys():
                parameters['win_length'] = len(self.data)
            if 'hop_length' in parameters.keys():
                parameters['hop_length'] = len(self.data)
            if 'n_fft' in parameters.keys():
                parameters['n_fft'] = len(self.data)
            self.feature_full(function(**parameters), function.__name__, stats)
        else:
            raise ValueError("The desired statistics has not been implemented yet!")


def get_feature_name(ft_types: list, n_chunks=None):
    # define feature neame based on what function and what statistics are used
    # ft_types = [['spectral_centroid', 'full'], ['spectral_bandwidth', 'variance'], ..]
    feat_names = []
    for ft, stat in ft_types:
        if ft == 'duration':
            feat_names.append('duration')
        else:
            if stat == 'full':
                feat_names.append(f"{ft}_chunk1of1")
            elif stat == 'mean' or stat == 'variance':
                feat_names.append(f"{ft}_{stat}_chunk{n_chunks}of{n_chunks}")
            elif 'chunk' in stat and stat != 'one_chunk':
                assert n_chunks is not None, "Please provide number of chunks desired!"
                for ch in range(n_chunks):
                    feat_names.append(f"{ft}_{stat[:-6]}_chunk{ch+1}of{n_chunks}")
            elif 'delta' in stat:
                assert n_chunks is not None, "Please provide number of chunks desired!"
                if 'delta2' not in stat:
                    if n_chunks == 1:
                        feat_names.append(f"{ft}_delta_chunk1of1")
                    else:
                        n_deltas = n_chunks - 1
                        for d in range(n_deltas):
                            feat_names.append(f"{ft}_delta_chunk{d + 1}of{n_deltas}")
                else:
                    if n_chunks == 1:
                        feat_names.append(f"{ft}_delta_chunk1of1")
                    else:
                        n_deltas = n_chunks - 2
                        for d in range(n_deltas):
                            feat_names.append(f"{ft}_delta2{d + 1}of{n_deltas}")
            elif stat == 'delta_full':
                feat_names.append(f"{ft}delta_chunk1of1")
    return feat_names
