import numpy as np
import tensorflow as tf
import pickle
import tensorflow
from tensorflow import keras
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

model = keras.models.load_model('saved_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

Y_columns = ['Contract', 'Criminal', 'Family', 'Property', 'Tort_1', 'Tort_2']
Y_columns = sorted(Y_columns)

ps = PorterStemmer()
sequence_len = 20

def model_predict(text):
    print('@ Sentence input:', text)

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    
    print(text)
    sequence = tokenizer.texts_to_sequences( [text] )[0]

    if (len(sequence) == 0):
        return Y_columns

    if len(sequence) < sequence_len:
        # sequence = [0] * (sequence_len - len(sequence)) + sequence
        num_repeats = (sequence_len // len(sequence)) + 1
        sequence = sequence * num_repeats
#     print('@ sequence:', sequence)

    sequences = []
    num_sequences = len(sequence) // sequence_len
    
    for i in range(num_sequences):
        start = sequence_len * i
        end = sequence_len * (i + 1)
        sequences.append(sequence[start:end])
    
    y_prob = model.predict(sequences)
    y_avg_prob = list(np.average(y_prob, axis=0))

    y_avg_prob_map = { Y_columns[i] : y_avg_prob[i] for i in range(len(y_avg_prob)) }
#     print('Map:', y_avg_prob_map)

    y_avg_prob_map_sorted = {k: v for k, v in sorted(y_avg_prob_map.items(), key=lambda x: x[1], reverse=True)}
    print('Sorted Map:', y_avg_prob_map_sorted)
    
    category_list = list(y_avg_prob_map_sorted.keys())
#     print('Category list: ', category_list)

    return category_list

# test_sentences = [
#     'Divorcing my wife',
#     'Abused by my husband',
#     'I want to create a will',
#     'Mental health dropping',
#     'I wish to adopt a child',
    
#     'I have an employment issue',
#     'Want to Claim insurance',
#     'I took out a large loan',
#     'I need some help on an investment I made',
#     'I bought a product and need to use lemon law',
    
#     'I intend to buy a BTO house',
#     'paying my mortgage',
#     'I had a huge fight with my tenant. I am not sure what I should do. He keeps screaming at me and does not reply me',
#     'I got sued for a copyright',
#     'I had a will estate planning issue',
    
#     'I was harassed by some stranger on the street',
#     'I received a defamation letter',
#     'I have issues with privacy',
#     'PDPA is a serious concern my company faces',
#     'My neighbor keeps banging my door',
    
#     'I wish to make a personal injury claim',
#     'I was threatened by the man earlier',
#     'My employer neglected my well-being',
#     'I got into a traffic accident',
#     'My doctor did not prescribe the right medicine',
# ]

# for sentence in test_sentences:
#     model_predict(sentence)