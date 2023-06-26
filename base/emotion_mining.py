import operator

from text_preprocessor import pre_process

data_dir = 'data'
emotion_seed_files = ['anger.txt', 'disgust.txt', 'fear.txt', 'joy.txt',
                      'sadness.txt', 'surprise.txt']

emotion_seed_map = {}

for seed_file in emotion_seed_files:
    lines = [line.replace("\n", "") for line in open("data/seeds/" + seed_file)]
    emotion_seed_map[seed_file.replace(".txt", "").strip()] = lines


def get_emotion_distribution(facebook_post):
    processed_tokens = pre_process(facebook_post)
    emotion_map = {}
    for tkn in processed_tokens:
        for emotion in emotion_seed_map:
            if tkn in emotion_seed_map[emotion]:
                # TODO : use negation
                emotion_map[emotion] = emotion_map.setdefault(emotion, 0) + 1
    if len(emotion_map) > 0:
        return max(emotion_map.iteritems(), key=operator.itemgetter(1))[0]
    else:
        return "unknown"


import pandas as pd

if __name__ == '__main__':
    data_set = pd.read_csv('data/Sentenses.csv')
    labelled = open('data/labelled.csv', 'w')
    un_labelled = open('data/un_labelled.csv', 'w')
    labelled.write("Sentences,label \n")
    un_labelled.write("Sentences \n")
    for index, row in data_set.iterrows():
        print(index)
        sent = row['Sentences']
        result = get_emotion_distribution(sent)
        if result == "unknown":
            un_labelled.write(sent + "\n")
        else:
            labelled.write(sent + "," + result + "\n")
    labelled.close()
    un_labelled.close()
