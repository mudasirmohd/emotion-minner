__author__ = "Shah Muzaffar"
import json
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from nltk import word_tokenize
import nltk

nltk.download("punkt")
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words='english',
                             lowercase=True, token_pattern='[a-zA-Z0-9]+', strip_accents='unicode',
                             tokenizer=word_tokenize)


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

    return data_array, label_array


def write_class_map(class_file, label_map):
    with open(class_file, 'w') as fp:
        for label_name in label_map:
            mp = {'label_name': label_name, 'label_number': label_map[label_name]}
            fp.write(json.dumps(mp) + "\n")
    fp.close()


def write_labels(label_array, file_name):
    with open(file_name, 'w') as wr:
        wr.write(json.dumps(label_array))


def vectorize(tr_file, data_dir):
    data_file = data_dir + "/" + tr_file
    text_header = 'Sentences'
    target_field_header = 'label'
    skip_tags = ['Unknown', 'h', 'w', 'Tag']
    data, labels, = read_data(data_file, text_header, target_field_header, skip_tags)
    label_encode = LabelEncoder()
    all_labels = label_encode.fit_transform(labels)
    num_classes = np.unique(all_labels).shape[0]

    print("number of classes :  ", num_classes)
    mylist = list(xrange(num_classes))
    print(label_encode.inverse_transform(mylist))

    logging.info("pickling label encoder")
    pickle.dump(label_encode, open(data_dir + "label_encoder.pickle", "wb"))

    print "data loaded successfully , going to vectorize data"
    if len(labels) != len(data):
        print "exception occurred while loading data ::::::,size of data  set and label set un equal"
    else:
        vec_matrix = vectorizer.fit_transform(data)
        pickle.dump(vectorizer, open(data_dir + 'vectorizer.pickle', 'wb'))
        return vec_matrix, all_labels


if __name__ == '__main__':
    from classification_trainer import create_model

    directory = "../data/"
    training_file = directory + "labelled.csv"
    X, y = vectorize(tr_file=training_file, data_dir=directory)
    create_model(X, y, directory + '/model.dat')


