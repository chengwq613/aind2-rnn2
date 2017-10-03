import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    series_size = len(series)
    print(series_size)
    for start_i in range(0, (series_size-window_size), 1):
        end_i = start_i + window_size
        Input = series[start_i:end_i]
        Output = series[end_i]
        X.append(Input)
        y.append(Output)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape = (window_size,1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = [chr(i) for i in range(ord('a'),ord('z')+1)]
    valid = punctuation + letters
    
    for i in text:
        if i not in valid:
            text = text.replace(i,' ') 
        
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    text_size = len(text)

    for start_i in range(0, (text_size-window_size), step_size):
        end_i = start_i + window_size
        Input = text[start_i:end_i]
        Output = text[end_i]
        inputs.append(Input)
        outputs.append(Output)

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape = (window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
