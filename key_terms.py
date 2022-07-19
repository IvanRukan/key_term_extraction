import nltk
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer


def xml_extractor():
    tree = etree.parse('A:\\downloads\\news.xml')
    root = tree.getroot()
    return root[0]


def tokenizer(root):
    vectorizer = TfidfVectorizer()

    stop_words = stopwords.words('english') + ['ha', 'wa', 'u', 'a']
    lemmatizer = WordNetLemmatizer()
    nouns = []
    result = []
    titles = []
    for each in root:
        text = each[1].text
        tokenized_words = word_tokenize(text.lower())
        tokenized_words = sorted(tokenized_words, reverse=True)
        tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
        tokenized_words = [word.rstrip(punctuation) for word in tokenized_words if word not in stop_words]
        tokenized_words = [word for word in tokenized_words if word]
        tokenized_words = [word for word in tokenized_words if nltk.pos_tag([word])[0][1] == 'NN']
        nouns.append(' '.join(tokenized_words))
        titles.append(each[0].text)

    matrix = vectorizer.fit_transform(nouns)
    # test

    # test
    for row in matrix.toarray():
        new_what = list(zip(vectorizer.get_feature_names_out(), row))
        new_what = [each for each in new_what if each[1] != 0]
        values = [each[1] for each in new_what]

        values = sorted(values, reverse=True)
        words = []

        for i in range(5):
            temp = []
            for each in new_what:
                if each[1] == values[i]:
                    temp.append(each[0])
            temp = set(temp)
            temp = sorted(temp, reverse=True)
            words += temp
        used_words = []
        no_duplicates = []
        for word in words:
            if word in used_words:
                continue
            no_duplicates.append(word)
            used_words.append(word)

        if len(no_duplicates) > 5:
            while len(no_duplicates) != 5:
                no_duplicates.pop()
        result.append(no_duplicates)
    for title, content in zip(titles, result):
        print(title + ':')
        print(*content)
        print()


tokenizer(xml_extractor())

