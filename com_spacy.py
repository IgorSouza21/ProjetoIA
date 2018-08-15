import spacy
from preprocessamento import dicionario


nlp = spacy.load('en')


def tokenizarspacy(docs):
    stems = []
    tamanhos = []
    for doc in docs:
        t = tokenStringspacy(doc[0])
        tamanhos.append(len(t))
        for x in t:
            stems.append(x)
    return stems, tamanhos


def tokenStringspacy(string):
    i = nlp(string)
    s = [token.lemma_ for token in i if (not token.is_stop) and (
                (not token.is_punct or token.text == '!') and token.is_alpha and token.pos_ is not 'PRON')]
    return s


def tokenDocspacy(doc):
    i = nlp(doc[0])
    s = [token.lemma_ for token in i if (not token.is_stop) and (
            (not token.is_punct or token.text == '!') and token.is_alpha and token.pos_ is not 'PRON')]
    s.append(doc[1])
    return s


def dicionariospacy(docs):
    lemmas, tamanhos = tokenizarspacy(docs)
    fs, df = dicionario(lemmas, 0)

    return fs, df, tamanhos
