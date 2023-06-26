
__author__ = "Shah Muzaffar"

from sklearn import svm

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, \
    accuracy_score, f1_score


def create_model(training_matrix, labels, model_path):
    x_train, x_test, y_train, y_test = train_test_split(training_matrix, labels, test_size=0.02, random_state=42)
    clf = svm.SVC()

    clf.fit(training_matrix, labels)
    print("going to evaluate model")

    prediction = clf.predict(x_test)
    print 'accuracy_score:   ', accuracy_score(y_test, prediction)
    print 'f1_score:   ', f1_score(y_test, prediction, average=None)
    print 'recall_score:   ', recall_score(y_test, prediction, average=None)
    print 'precision_score:', precision_score(y_test, prediction, average=None)
    print 'confusion_matrix:', confusion_matrix(y_test, prediction)

    print("going to train on all data")

    print "training complete going to save model..."
    pickle.dump(clf, open(model_path, 'wb'))
    print "model saved at path + " + model_path



