import tensorflow as tf
import numpy as np
import pickle
from scipy import io


class Infer:
    def __init__(self, dimension=32, vocab_size=32889, save_dir='checkpoints/'):
        self.save_dir = save_dir
        self.dictionary = pickle.load(open(save_dir + "dict.pickle"))
        self.label_encoder = pickle.load(open(save_dir + "label_encoder.pickle"))
        self.vectors = io.mmread(save_dir + "matrix.mtx")
        self.dimension = dimension
        self.vocab_size = vocab_size

    def classify(self, post):
        batch = 1
        batch_x = np.random.rand(self.dimension, batch, self.vocab_size)
        emb_data = np.zeros((self.dimension, self.vocab_size), dtype=np.float32)

        for _, text in enumerate(post.split()):
            # if the word got in the vocab
            try:
                emb_data[:, self.dictionary.index(text)] += self.vectors[self.dictionary.index(text), :]

            except:
                continue
        batch_x[:, 0, :] = emb_data[:, :]

        graph = tf.get_default_graph()
        with graph.as_default():
            sess = tf.Session(graph=graph)
            saver = tf.train.import_meta_graph(self.save_dir + "best_validation-2.meta")
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
            x = graph.get_tensor_by_name("input:0")
            y_pred = graph.get_tensor_by_name("y_pred:0")
            feed_dict_testing = {x: batch_x}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)
            y_pred_cls = tf.argmax(result, axis=1)
            z = (sess.run(y_pred_cls)[0])
            x_max = tf.reduce_max(result, reduction_indices=[1])
            probability_score = sess.run(x_max)
            return self.label_encoder.inverse_transform(z), probability_score
