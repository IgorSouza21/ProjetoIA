from TratamentosNaiveBayes import pasta
from avaliacao import Type
from naiveBayesV import crossValidationV
from naiveBayes import crossValidation
from jpretext import crossValidationJpretext


def crossBinNLTK(k, textos, polaridades, classes, biblioteca):
    r = crossValidationV(k, textos, polaridades, classes, biblioteca, Type.BIN)
    p = organiza(r)
    print(p)
    arq = open(pasta + str(k) + '-folds com tratamento binário NLTK.txt', 'w')
    arq.writelines(p)

    return arq


def crossTFnormalizadoJpretext(k, classes):
    r = crossValidationJpretext(k, classes, 'textosTF')
    p = organiza(r)
    print(p)
    arq = open(pasta + str(k) + '-folds com Jpretext TF.txt', 'w')
    arq.writelines(p)

    return arq


def crossBinSpacy(k, textos, polaridades, classes, biblioteca):
    r = crossValidationV(k, textos, polaridades, classes, biblioteca, Type.BIN)
    p = organiza(r)
    print(p)
    arq = open(pasta + str(k) + '-folds com tratamento binário Spacy.txt', 'w')
    arq.writelines(p)

    return arq


def crossSemTratamento(k, textos, polaridades, classes):
    r = crossValidation(k, textos, polaridades, classes)
    p = organiza(r)
    print(p)
    arq = open(pasta + str(k) + '-folds sem tratamento.txt', 'w')
    arq.writelines(p)

    return arq


def organiza(r):
    p = []
    p.append('acuracia: ' + str(r[0] * 100) + '%\n')
    p.append('erro: ' + str(r[1] * 100) + '%\n')
    p.append('precisao POS: ' + str(r[2] * 100) + '%\n')
    p.append('precisao NEG: ' + str(r[3] * 100) + '%\n')
    p.append('precisao NEUTRO: ' + str(r[4] * 100) + '%\n')
    p.append('recall POS: ' + str(r[5] * 100) + '%\n')
    p.append('recall NEG: ' + str(r[6] * 100) + '%\n')
    p.append('recall NEUTRO: ' + str(r[7] * 100) + '%\n')
    p.append('f-measure: ' + str(r[8] * 100) + '%\n')

    return p