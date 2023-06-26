import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import more_itertools
import logging
from collections import Counter
import itertools
import tensorflow as tf
import numpy as np
import collections
import random
import time
from sklearn.model_selection import train_test_split
from rnn_model import Model
from WordModel import WordModel
import pickle
from sklearn import metrics
from scipy import io

import os
import gc
from sklearn.utils import shuffle


def clean_str(s):
    try:
        s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
        s = re.sub(r" : ", ":", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\(", " \( ", s)
        s = re.sub(r"\)", " \) ", s)
        s = re.sub(r"\?", " \? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip().lower()
    except:
        return ""


def read_data(filename, text_header, target_field_header, skip_tags):
    data_set = pd.read_csv(filename)
    print(data_set.shape, "before filtering")
    data_set = data_set[pd.notnull(data_set[target_field_header])]
    for skip_tag in skip_tags:
        try:
            data_set = data_set[data_set.Tag != skip_tag]
        except Exception:
            pass
    print(data_set.shape, "after filtering")
    label_array = data_set[target_field_header]
    label_array = [a.lower() for a in label_array]
    data_array = data_set[text_header]
    data_array = [clean_str(line) for line in data_array]

    rows = data_set.shape[0]
    logging.info('there are', rows, 'total rows')
    all_words_ = list(
        more_itertools.flatten([[word for word in clean_str(sentence).split()] for sentence in data_array]))

    return all_words_, data_array, label_array


def generate_batch_skip_gram(words, batch_size, num_skips, skip_window):
    data_index = 0

    # check batch_size able to convert into number of skip in skip-grams method
    assert batch_size % num_skips == 0

    assert num_skips <= 2 * skip_window

    # create batch for model input
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1

    # a buffer to placed skip-grams sentence
    buffer = collections.deque(maxlen=span)
    for i in range(span):
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        for j in range(num_skips):

            while target in targets_to_avoid:
                # random a word from the sentence
                # if random word still a word already chosen, simply keep looping
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)

    return batch, labels


def build_dataset(words):
    word_counts = Counter(itertools.chain(words))
    vocabulary_list = [word[0] for word in word_counts.most_common()]
    vocab_dict = {word: index for index, word in enumerate(vocabulary_list)}
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}
    word_indices_lst = [vocab_dict[word] for word in words]
    return word_indices_lst, vocabulary_list, rev_vocab_dict


def train_word_embending_model(batch_size, dimension, skip_window, num_skips, num_itters,
                               word_indices_data, vocab_size):
    sess = tf.InteractiveSession()
    model = WordModel(batch_size, dimension, 0.01, vocab_size)
    sess.run(tf.global_variables_initializer())
    for step in range(num_itters):
        new_time = time.time()
        batch_inputs, batch_labels = generate_batch_skip_gram(word_indices_data, batch_size, num_skips, skip_window)
        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)

        if ((step + 1) % 1000) == 0:
            print('epoch: ', step + 1, ', loss: ', loss, ', speed: ', time.time() - new_time)

    tf.reset_default_graph()
    vectors = model.normalized_embeddings.eval()
    return vectors


def train_rnn(X_train, Y_train, X_test, Y_test, dimension, vectors, vocab_size, num_classes, dictionary, save_dir):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    num_layers = 2
    size_layer = 256
    learning_rate = 0.001
    epoch = 50
    batch = 30
    X_test = X_test[:32]
    Y_test = Y_test[:32]
    test_batch = len(X_test)
    print ("test batch size is {} ".format(test_batch))
    model = Model(num_layers, size_layer, vocab_size, num_classes, learning_rate)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    save_path = os.path.join(save_dir, 'best_validation-2')
    ACC_TRAIN, ACC_TEST, LOST = [], [], []
    for i in range(epoch):
        total_cost = 0
        total_accuracy = 0
        last_time = time.time()

        for n in range(0, (len(X_train) // batch) * batch, batch):
            batch_x = np.zeros((dimension, batch, vocab_size))
            batch_y = np.zeros((batch, num_classes))
            for k in range(batch):
                emb_data = np.zeros((dimension, vocab_size), dtype=np.float32)
                for _, text in enumerate(X_train[n + k].split()):
                    # if the word got in the vocab
                    try:
                        emb_data[:, dictionary.index(text)] += vectors[dictionary.index(text), :]

                    except:
                        continue

                batch_y[k, int(Y_train[n + k])] = 1.0

                batch_x[:, k, :] = emb_data[:, :]
            loss, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y})
            total_accuracy += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
            total_cost += loss
        gc.collect()
        saver.save(sess=sess, save_path=save_path)

        total_cost /= (len(X_train) // batch)
        total_accuracy /= (len(X_train) // batch)
        times = (time.time() - last_time) / (len(X_train) // batch)

        ACC_TRAIN.append(total_accuracy)
        LOST.append(total_cost)

        if i % 5 == 0:

            print('epoch: ', i, ', loss: ', total_cost, ', accuracy train: ', total_accuracy, 's / batch: ', times)

            batch_x = np.zeros((dimension, test_batch, vocab_size))
            batch_y = np.zeros((test_batch, num_classes))

            for k in range(test_batch):
                emb_data = np.zeros((dimension, vocab_size), dtype=np.float32)
                for _, text in enumerate(X_test[k].split()):
                    try:
                        emb_data[:, dictionary.index(text)] += vectors[dictionary[text], :]
                    except:
                        continue

                batch_y[k, int(Y_test[k])] = 1.0
                batch_x[:, k, :] = emb_data[:, :]

            testing_acc, logits = sess.run([model.accuracy, tf.cast(tf.argmax(model.logits, 1), tf.int32)],
                                           feed_dict={model.X: batch_x, model.Y: batch_y})
            print ('testing accuracy: ', testing_acc)
            ACC_TEST.append(testing_acc)
            print (metrics.classification_report(Y_test, logits))


def main(data_file, save_dir, text_header, target_field_header, skip_tags):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = 30
    dimension = 32
    skip_window = 1
    num_skips = 2
    num_itterations = 5000

    all_words, all_data, labels = read_data(data_file, text_header, target_field_header, skip_tags)
    label_encode = LabelEncoder()
    all_labels = label_encode.fit_transform(labels)
    num_classes = np.unique(all_labels).shape[0]

    print("number of classes :  ", num_classes)
    mylist = list(xrange(num_classes))
    print(label_encode.inverse_transform(mylist))

    logging.info("pickling label encoder")
    pickle.dump(label_encode, open(save_dir + "label_encoder.pickle", "wb"))
    word_indices_data, dictionary, reverse_dictionary = build_dataset(all_words)
    print('Creating Word2Vec model.')
    vocab_size = len(dictionary)
    print("vocab size is {}".format(vocab_size))

    logging.info("pickling dictionary")
    pickle.dump(dictionary, open(save_dir + "dict.pickle", "wb"))

    vectors = train_word_embending_model(batch_size=batch_size, dimension=dimension, skip_window=skip_window,
                                         num_skips=num_skips, num_itters=num_itterations,
                                         word_indices_data=word_indices_data, vocab_size=vocab_size)
    io.mmwrite(save_dir + "matrix.mtx", vectors)
    X, y = shuffle(all_data, all_labels)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.001)
    print("shape of vectors : ", vectors.shape)
    print("len of X_train : ", len(X_train))
    print("len of Y_train : ", len(Y_train))
    print("len(X_train) // batch: ", len(X_train) // 30)
    train_rnn(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, dimension=dimension, vectors=vectors,
              num_classes=num_classes, dictionary=dictionary, save_dir=save_dir, vocab_size=vocab_size)


if __name__ == '__main__':
    data_file = '../data/merged.csv'
    save_dir = '../data/checkpoints/'
    text_header = 'Sentences'
    target_field_header = 'label'
    skip_tags = ['Unknown', 'h', 'w', 'Tag', 'label']

    main(data_file, save_dir, text_header, target_field_header, skip_tags)
