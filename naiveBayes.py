import math
import random
from naiveBayesV import holdout, pp
import avaliacao


def vocabulario(D):
    V = []
    for d in D:
        t = d[0].split()
        for word in t:
            if word not in V:
                V.append(word)
    return V


def trainNaiveBayes(D, C):
    loglikelihood = {}
    logprior = {}
    bigdoc = {}
    V = vocabulario(D)
    Ndoc = len(D)
    for c in C:
        bigdoc[c] = []
        loglikelihood[c] = {}
        Nc = 0
        for d in D:
            if d[1] == c:
                Nc = Nc + 1
                bigdoc[c].append(d[0])
        logprior[c] = math.log2(Nc/Ndoc)
        for w in V:
            loglikelihood[c][w] = 0
            contWC = 0
            contC = 0
            for doc in bigdoc[c]:
                contWC = contWC + doc.count(w)
                contC = contC + len(doc.split())
            loglikelihood[c][w] = math.log2((contWC + 1) / (contC + len(V)))
    return logprior, loglikelihood, V


def testeNaiveBayes(testedoc, logprior, loglikelihood, C, V):
    teste = testedoc[0].split()
    sum = {}
    for c in C:
        sum[c] = logprior[c]
        for word in teste:
            if word in V:
                sum[c] = sum[c] + loglikelihood[c][word]
    m = max(sum.values())
    for key in sum:
        if m is sum[key]:
            return key


def matrizConfusao(teste, logprior, loglikelihood, classes, V):
    #         VP  FN FT  FP VN  FT  FP  FN VT
    matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t in teste:
        classe = testeNaiveBayes(t, logprior, loglikelihood, classes, V)
        if t[1] == 1 and classe == 1:
            matriz[0][0] += 1
        elif t[1] == 1 and classe == -1:
            matriz[0][1] += 1
        elif t[1] == 1 and classe == 0:
            matriz[0][2] += 1
        elif t[1] == -1 and classe == 1:
            matriz[1][0] += 1
        elif t[1] == -1 and classe == -1:
            matriz[1][1] += 1
        elif t[1] == -1 and classe == 0:
            matriz[1][2] += 1
        elif t[1] == 0 and classe == 1:
            matriz[2][0] += 1
        elif t[1] == 0 and classe == -1:
            matriz[2][1] += 1
        elif t[1] == 0 and classe == 0:
            matriz[2][2] += 1

    return matriz


def naiveBayes(textos, polaridades, classes):
    treino, teste, quantTreino = holdout(textos, polaridades)
    random.shuffle(treino)

    logprior, loglikelihood, V = trainNaiveBayes(treino, classes)
    matriz = matrizConfusao(teste, logprior, loglikelihood, classes, V)
    s1 = avaliacao.printMatriz(matriz)
    s2, p = avaliacao.resultados(matriz)

    return s1, s2


def crossValidation(k, textos, polaridades, classes):
    tudo = []
    for i in range(len(textos)):
        tudo.append(pp.document(textos[i], polaridades[i]))

    random.shuffle(tudo)

    fold = math.floor(len(tudo)/k)
    resultados = []

    for i in range(k):
        print('fold -> ' + str(i+1) + '/' + str(k))
        treino = tudo[0:fold * i][0:fold * i] + tudo[fold * (i + 1):len(tudo)][fold * (i + 1):len(tudo)]
        logprior, loglikelihood, V = trainNaiveBayes(treino, classes)
        matriz = matrizConfusao(tudo[fold * i:fold * (i + 1)], logprior, loglikelihood, classes, V)
        descarta, lista = avaliacao.resultados(matriz)
        resultados.append(lista)

    acuracia = []
    erro = []
    precisaoPOS = []
    precisaoNEG = []
    precisaoNEU = []
    recallPOS = []
    recallNEG = []
    recallNEU = []
    f_measure = []
    for res in resultados:
        for j in range(len(res)):
            if j == 0:
                acuracia.append(res[j])
            elif j == 1:
                erro.append(res[j])
            elif j == 2:
                precisaoPOS.append(res[j])
            elif j == 3:
                precisaoNEG.append(res[j])
            elif j == 4:
                precisaoNEU.append(res[j])
            elif j == 5:
                recallPOS.append(res[j])
            elif j == 6:
                recallNEG.append(res[j])
            elif j == 7:
                recallNEU.append(res[j])
            else:
                f_measure.append(res[j])

    return [sum(acuracia)/k, sum(erro)/k, sum(precisaoPOS)/k, sum(precisaoNEG)/k,
            sum(precisaoNEU) / k, sum(recallPOS)/k, sum(recallNEG)/k, sum(recallNEU)/k, sum(f_measure)/k]