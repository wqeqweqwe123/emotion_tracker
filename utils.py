import os

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def ensure_models_dir(path='models'):
    os.makedirs(path, exist_ok=True)


def labels():
    return EMOTIONS
