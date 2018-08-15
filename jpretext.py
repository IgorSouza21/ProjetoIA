import os
import csv
import lerXmlInputProjeto
from naiveBayesV import *


def naiveBayesJpretext(classes, string, elevar=False):
    BOW = trataJpretext(string)
    quantTreino = math.floor((2 * len(BOW)) / 3)
    matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    logprior, loglikelihood = trainNaiveBayes(BOW[:quantTreino], classes)

    for i in range(quantTreino, len(BOW)):
        previsao = testeNaiveBayes(BOW[i], logprior, loglikelihood, classes, elevar)
        matrizConfusao(BOW[i][-1], previsao, matriz)
    p = printMatriz(matriz)
    p2, x = resultados(matriz)

    return p, p2


def jpretext():
    pasta = 'jpretext/textos'
    if not os.path.exists(pasta):
        os.mkdir(pasta)
        textos = lerXmlInputProjeto.selecionarTodosTextos()
        polaridades = lerXmlInputProjeto.selecionarPolaridades()
        pos = []
        neg = []
        neutro = []
        for i in range(len(textos)):
            if polaridades[i] == 1:
                pos.append(textos[i])
            elif polaridades[i] == -1:
                neg.append(textos[i])
            else:
                neutro.append(textos[i])

        for i in range(len(pos)):
            arq = open(pasta + "/pos." + str(i) + ".txt", "w")
            arq.write(pos[i])
            arq.close()
        for i in range(len(neg)):
            arq = open(pasta + "/neg." + str(i) + ".txt", "w")
            arq.write(neg[i])
            arq.close()
        for i in range(len(neutro)):
            arq = open(pasta + "/neutro." + str(i) + ".txt", "w")
            arq.write(neutro[i])
            arq.close()


def trataJpretext(string):
    arq = open('jpretext/' + string + '.csv', newline='')
    p = csv.reader(arq, delimiter=',')
    l = []
    for x in p:
        l.append(x[0].split(';'))

    BOW = []
    for i in range(1, len(l)):
        lista = l[i][1:]
        if l[i][0].count('pos') > 0:
            lista.append(1)
        elif l[i][0].count('neg') > 0:
            lista.append(-1)
        else:
            lista.append(0)
        BOW.append(lista)

    for i in range(len(BOW)):
        for j in range(len(BOW[i])):
            BOW[i][j] = float(BOW[i][j])

    return BOW
