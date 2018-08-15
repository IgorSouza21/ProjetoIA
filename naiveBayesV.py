import math
from com_nltk import *
from com_spacy import *
import random
import preprocessamento as pp
from avaliacao import *


def trainNaiveBayes(D, C):
    loglikelihood = {}
    logprior = {}
    bigdoc = {}
    tamDicionario = len(D[0]) - 1
    Ndoc = len(D)
    for c in C:
        bigdoc[c] = []
        loglikelihood[c] = {}
        Nc = 0
        for d in D:
            if d[-1] == c:
                Nc = Nc + 1
                bigdoc[c].append(d)
        logprior[c] = math.log2(Nc/Ndoc)
        for w in range(tamDicionario):
            loglikelihood[c][w] = 0
            contWC = 0
            contC = 0
            for doc in bigdoc[c]:
                contWC = contWC + doc[w]
                contC = contC + sum(doc)
            loglikelihood[c][w] = math.log2(((contWC + 1) / (contC + tamDicionario)))
    return logprior, loglikelihood


def testeNaiveBayes(testedoc, logprior, loglikelihood, C, elevar):
    sum = {}
    for c in C:
        sum[c] = logprior[c]
        for i in range(len(testedoc)-1):
            if testedoc[i] != 0:
                if elevar:
                    try:
                        #loglikelihood[c][i] = math.log2((math.pow(2, loglikelihood[c][i])) ** testedoc[i])
                        loglikelihood[c][i] = loglikelihood[c][i] * testedoc[i]
                    except ValueError:
                        loglikelihood[c][i] = 0
                sum[c] = sum[c] + loglikelihood[c][i]
    m = max(sum.values())
    for key in sum:
        if m is sum[key]:
            return key


def matrizConfusao(classeReal, classePrevista, matriz):
    if classeReal == 1 and classePrevista == 1:
        matriz[0][0] += 1
    elif classeReal == 1 and classePrevista == -1:
        matriz[0][1] += 1
    elif classeReal == 1 and classePrevista == 0:
        matriz[0][2] += 1
    elif classeReal == -1 and classePrevista == 1:
        matriz[1][0] += 1
    elif classeReal == -1 and classePrevista == -1:
        matriz[1][1] += 1
    elif classeReal == -1 and classePrevista == 0:
        matriz[1][2] += 1
    elif classeReal == 0 and classePrevista == 1:
        matriz[2][0] += 1
    elif classeReal == 0 and classePrevista == -1:
        matriz[2][1] += 1
    elif classeReal == 0 and classePrevista == 0:
        matriz[2][2] += 1


def matrizBOW(docs, biblioteca, score, normalizar):
    dicionario = []
    df = []
    tamanhos = []
    if biblioteca == Type.NLTK:
        dicionario, df, tamanhos = dicionarionltk(docs)
    elif biblioteca == Type.SPACY:
        dicionario, df, tamanhos = dicionariospacy(docs)

    ss = []
    for k in range(len(docs)):
        ds = []
        if biblioteca == Type.NLTK:
            ds = tokenStringnltk(docs[k][0])
        elif biblioteca == Type.SPACY:
            ds = tokenStringspacy(docs[k][0])
        linha = []
        for word in range(len(dicionario)):
            tf = ds.count(dicionario[word])
            if score == Type.BIN:
                if tf > 0:
                    linha.append(1)
                else:
                    linha.append(0)
            elif score == Type.TF:
                if normalizar:
                    try:
                        linha.append(tf/tamanhos[k])
                    except ArithmeticError:
                        linha.append(tf)
                else:
                    linha.append(tf)
            elif score == Type.TFIDF:
                if normalizar:
                    try:
                        x = tf/tamanhos[k]
                        linha.append(x/df[word])
                    except ArithmeticError:
                        linha.append(tf/df[word])
                else:
                    linha.append(tf/df[word])
        linha.append(docs[k][1])
        ss.append(linha)
    return ss


def naiveBayesV(textos, polaridades, classes, biblioteca, score, normalizar=False, elevar=False):
    treino, teste, quantTreino = holdout(textos, polaridades)
    random.shuffle(treino)
    BOW = matrizBOW(treino + teste, biblioteca, score, normalizar)
    matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    logprior, loglikelihood = trainNaiveBayes(BOW[:quantTreino], classes)
    for i in range(quantTreino, len(BOW)):
        previsao = testeNaiveBayes(BOW[i], logprior, loglikelihood, classes, elevar)
        matrizConfusao(BOW[i][-1], previsao, matriz)
    p = printMatriz(matriz)
    p2, x = resultados(matriz)

    return p, p2


def holdout(textos, polaridades):
    total = len(textos)
    separado = pp.separaPositivosNegativosNeutros(textos, polaridades)
    treino = []
    teste = []
    per = pp.estratificacao(separado, total)
    quantTreino = math.floor((2/3)*total)

    train = []
    l = len(separado)
    for i in range(l):
        cont = math.floor(quantTreino * per[i])
        if i != len(separado)-1:
            train.append(cont+1)
        else:
            train.append(cont)

    test = []
    for i in range(len(train)):
        test.append(len(separado[i]) - train[i])

    for i in range(len(train)):
        for valor in range(train[i]):
            treino.append(separado[i][valor])

    for i in range(len(test)):
        for valor in range(test[i]):
            teste.append(separado[i][valor])

    return treino, teste, quantTreino


def crossValidation(k, textos, polaridades, classes, biblioteca, score, normalizar=False, elevar=False):
    tudo = []
    for i in range(len(textos)):
        tudo.append(pp.document(textos[i], polaridades[i]))

    BOW = matrizBOW(tudo, biblioteca, score, normalizar)

    fold = math.floor(len(BOW)/k)
    resultados = []

    for i in range(k):
        logprior, loglikelihood = trainNaiveBayes(BOW[0:fold*i]+BOW[fold*(i+1):len(BOW)], classes)
        resultados.append(teste(logprior, loglikelihood, classes, BOW[fold*k:fold*(k+1)], elevar))

    acuracia = []
    erro = []
    precisao = []
    recall = []
    f_measure = []
    for res in resultados:
        for j in range(len(res)):
            if j == 0:
                acuracia.append(res[j])
            elif j == 1:
                erro.append(res[j])
            elif j == 2:
                precisao.append(res[j])
            elif j == 3:
                recall.append(res[j])
            else:
                f_measure.append(res[j])

    return [sum(acuracia)/k, sum(erro)/k. sum(precisao)/k. sum(recall)/k, sum(f_measure)/k]


def teste(logprior, loglikelihood, classes, docs, elevar):
    matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for testdoc in docs:
        previsao = testeNaiveBayes(testdoc, logprior, loglikelihood, classes, elevar)
        matrizConfusao(testdoc[-1], previsao, matriz)

    p2, results = resultados(matriz)

    return results
