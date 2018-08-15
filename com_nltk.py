from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocessamento import dicionario


def docTokenized(doc):
    docs = []
    di = []
    for d in doc:
        di = tokenStringnltk(d[0])
        di.append(d[1])
        docs.append(di)

    return docs


def tokenStringnltk(string):
    pontuacao = [',', '.', ':', ';', '/', '?', '[', ']', '(', ')', '{', '}', '#', '$', '%', '-', '@', '!']
    stopWords = stopwords.words('english')
    ps = SnowballStemmer('english')
    s = []
    tokens = word_tokenize(string)
    for w in tokens:
        if w not in s and w not in pontuacao and w not in stopWords:
            s.append(ps.stem(w))
    return s


def tokenDocnltk(doc):
    pontuacao = [',', '.', ':', ';', '/', '?', '[', ']', '(', ')', '{', '}', '#', '$', '%', '-', '@', '!']
    stopWords = stopwords.words('english')
    ps = SnowballStemmer('english')
    s = []
    tokens = word_tokenize(doc[0])
    for w in tokens:
        if w not in s and w not in pontuacao and w not in stopWords:
            s.append(ps.stem(w))
    s.append(doc[1])
    return s


def tokenizarnltk(docs):
    pontuacao = [',', '.', ':', ';', '/', '?', '[', ']', '(', ')', '{', '}', '#', '$', '%', '-', '@', '!']
    stopWords = stopwords.words('english')
    ps = SnowballStemmer('english')
    stems = []
    tamanhos = []
    k = 0

    for t in range(len(docs)):
        tokens = word_tokenize(docs[t][0])
        tamanhos.append(0)
        for w in tokens:
            if w not in pontuacao and w not in stopWords and w.isalpha():
                stems.append(ps.stem(w))
                tamanhos[k] += 1
        k += 1

    return stems, tamanhos


def dicionarionltk(docs):
    stems, tamanhos = tokenizarnltk(docs)
    fs, df = dicionario(stems, 0)

    return fs, df, tamanhos
