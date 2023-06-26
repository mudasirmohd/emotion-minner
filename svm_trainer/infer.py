import pickle

model = pickle.load(open("../data/model.dat"))
vectorizer = pickle.load(open("../data/vectorizer.pickle"))
label_encoder = pickle.load(open("../data/label_encoder.pickle"))

lines = [line.replace("\n", "") for line in open('../data/un_labelled.csv')]
with open("../data/new_labelled.csv", 'w') as wr:
    wr.write("Sentences,label\n")

    for line in lines:
        vector = vectorizer.transform([line])
        label = label_encoder.inverse_transform(model.predict(vector[0]))[0]
        wr.write(line + "," + str(label) + "\n")
