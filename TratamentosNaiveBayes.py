from jpretext import naiveBayesJpretext
from naiveBayes import naiveBayes
from naiveBayesV import *

pasta = 'resultados/'


def semTratamento(textos, polaridades, classes):
    s1, s2 = naiveBayes(textos, polaridades, classes)
    arq = open(pasta + 'resultados sem tratamento.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratBinNLTK(textos, polaridades, classes, biblioteca1):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca1, Type.BIN)
    arq = open(pasta + 'resultados com tratamento binário NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFNLTK(textos, polaridades, classes, biblioteca1):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca1, Type.TF)
    arq = open(pasta + 'resultados com tratamento TF NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFnormalizadoNLTK(textos, polaridades, classes, biblioteca1):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca1, Type.TF, True)
    arq = open(pasta + 'resultados com tratamento TF normalizado NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFNLTK(textos, polaridades, classes, biblioteca1):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca1, Type.TFIDF)
    arq = open(pasta + 'resultados com tratamento TFIDF NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFnormalizadoNLTK(textos, polaridades, classes, biblioteca1):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca1, Type.TFIDF, True)
    arq = open(pasta + 'resultados com tratamento TFIDF normalizado NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratBinSpacy(textos, polaridades, classes, biblioteca2):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca2, Type.BIN)
    arq = open(pasta + 'resultados com tratamento binário SPACY.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFSpacy(textos, polaridades, classes, biblioteca2):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca2, Type.TF)
    arq = open(pasta + 'resultados com tratamento TF SPACY.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFNormalizadoSpacy(textos, polaridades, classes, biblioteca2):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca2, Type.TF, True)
    arq = open(pasta + 'resultados com tratamento TF normalizado SPACY.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFSpacy(textos, polaridades, classes, biblioteca2):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca2, Type.TFIDF)
    arq = open(pasta + 'resultados com tratamento TFIDF SPACY.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFNormalizadoSpacy(textos, polaridades, classes, biblioteca2):
    s1, s2 = naiveBayesV(textos, polaridades, classes, biblioteca2, Type.TFIDF, True)
    arq = open(pasta + 'resultados com tratamento TFIDF normalizado SPACY.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFJpretext(classes):
    s1, s2 = naiveBayesJpretext(classes, 'textosTF')
    arq = open(pasta + 'resultados com Jpretext TF.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFJpretext(classes):
    s1, s2 = naiveBayesJpretext(classes, 'textosTFIDF')
    arq = open(pasta + 'resultados com Jpretext TFIDF.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFNaoNormalizadoJpretext(classes):
    s1, s2 = naiveBayesJpretext(classes, 'textosNaoNormalizadoTF')
    arq = open(pasta + 'resultados com Jpretext Não Normalizado TF.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


def tratTFIDFNaoNormalizadoJpretext(classes):
    s1, s2 = naiveBayesJpretext(classes, 'textosNaoNormalizadoTFIDF')
    arq = open(pasta + 'resultados com Jpretext Não Normalizado TFIDF.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()