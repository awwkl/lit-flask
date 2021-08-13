import numpy as np
import tensorflow as tf
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model('saved_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequence_len = 15

Y_columns = sorted(['Corporate', 'Criminal', 'Divorce', 'Legal_Procedures'])
map_idx_to_category = { i: Y_columns[i] for i in range(len(Y_columns)) }


def model_predict(sentence):
    print('@ Sentence input:', sentence)
    sequence = tokenizer.texts_to_sequences( [sentence] )[0]
    if len(sequence) < sequence_len:
        # sequence = [0] * (sequence_len - len(sequence)) + sequence
        num_repeats = (sequence_len // len(sequence)) + 1
        sequence = sequence * num_repeats
    print('@ sequence:', sequence)

    sequences = []
    num_sequences = len(sequence) // sequence_len
    
    for i in range(num_sequences):
        start = sequence_len * i
        end = sequence_len * (i + 1)
        sequences.append(sequence[start:end])
    
    y_prob = model.predict(sequences)
    y_avg_prob = list(np.average(y_prob, axis=0))

    y_avg_prob_map = { Y_columns[i] : y_avg_prob[i] for i in range(len(y_avg_prob)) if y_avg_prob[i] >= 0.1 }
    print('Map:', y_avg_prob_map)

    y_avg_prob_map_sorted = {k: v for k, v in sorted(y_avg_prob_map.items(), key=lambda x: x[1], reverse=True)}
    print('Sorted Map:', y_avg_prob_map_sorted)
    
    category_list = list(y_avg_prob_map_sorted.keys())
    print('Category list: ', category_list)

    return category_list
