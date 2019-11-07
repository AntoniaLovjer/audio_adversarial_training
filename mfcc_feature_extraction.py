def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2) # stepping at half window size
def extract_features(base_dir, sound_file_paths, sound_names ,bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for i, sound_file_path in enumerate(sound_file_paths):
        sound_file_full_path = os.path.join(base_dir, sound_file_path)
        sound_clip,s = librosa.load(sound_file_full_path)
        sound_clip = feature_normalize(sound_clip)
        label = sound_names[i]
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                # y: audio time series, sr: sampling rate, n_mfcc: number of MFCCs to return
                # librosa.feature.mfcc() function return numpy array with shape (bands, frames)
                # transpose since the model expects time axis(frames) come first
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T 
                mfccs.append(mfcc)
                labels.append(label)
    features = np.asarray(mfccs)
    return np.array(features), np.array(labels,dtype = np.str)

def one_hot_encode(labels):
    return np.asarray(pd.get_dummies(labels), dtype = np.float32)