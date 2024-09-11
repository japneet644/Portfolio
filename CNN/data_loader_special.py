import pickle
import gzip
import numpy as np

def load_data_wrapper():
    """
    Load and preprocess lattice data from a pickle file.

    Returns:
        np.array: Reshaped training inputs
    """
    # Open the pickle file
    # f = open('./DATA/zeromean8by8lattice.pkl', 'rb')
    f = open('./DATA/8by8lattice.pkl', 'rb')

    # Check if the file is gzip-compressed
    if f.read(2) == b'\x1f\x8b':
        f.seek(0)
        file_object = gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
        file_object = f

    # Load the data from the pickle file
    training_inputs = pickle.load(file_object, encoding="latin1")

    # Reshape the training inputs
    # Original shape: 32, 10000, 8, 8
    training_inputs = np.reshape(training_inputs, (320000, 8, 8, 1))

    return training_inputs
