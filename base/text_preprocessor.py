import nltk

nltk.download('stopwords')
import re
from nltk.corpus import stopwords


def pre_process(in_text):
    in_text = in_text.lower()
    in_text.replace("URL", "__URL__")

    #  This regex is for generalizing hash-tags
    # TODO : test it properly
    in_text.replace("^#", "hashtag")

    # This regex is for replacing more occurrences of one character
    #  in a row with the character itself
    # TODO: FIX IT as it is not working properly.
    cleaned_text = re.sub(r'([a-zA-Z])\\1{2,}', r'$1', in_text)
    cleaned_text = re.sub("\S*\d\S*", "", cleaned_text).strip()
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)

    tokens = nltk.word_tokenize(cleaned_text)
    return remove_stop_words(tokens)


def remove_stop_words(words):
    return [word for word in words if word not in set(stopwords.words('english'))]


if __name__ == '__main__':
    text = "this is a cat hg6t67u"
    print(pre_process(in_text=text))
