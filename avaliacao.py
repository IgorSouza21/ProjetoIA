from enum import Enum


class Type(Enum):
    NLTK = 0
    SPACY = 1
    JPRETEXT = 2
    TF = 3
    BIN = 4
    TFIDF = 5


def resultados(matriz):
    # VPOS[0][0] FNEG[0][1] FNEU[0][2]
    # FPOS[1][0] VNEG[1][1] FNEU[1][2]
    # FPOS[2][0] FNEG[2][1] VNEU[2][2]
    total = matriz[0][0] + matriz[0][1] + matriz[0][2] + matriz[1][0] + matriz[1][1] + matriz[1][2] + matriz[2][0] + \
            matriz[2][1] + matriz[2][2]

    acuracia = (matriz[0][0] + matriz[1][1] + matriz[2][2]) / total
    erro = 1 - acuracia
    try:
        precisaoPOS = matriz[0][0] / (matriz[0][0] + matriz[1][0] + matriz[2][0])
    except ArithmeticError:
        precisaoPOS = 0

    try:
        precisaoNEG = matriz[1][1] / (matriz[1][1] + matriz[0][1] + matriz[2][1])
    except ArithmeticError:
        precisaoNEG = 0

    try:
        precisaoNEU = matriz[2][2] / (matriz[2][2] + matriz[0][2] + matriz[1][2])
    except ArithmeticError:
        precisaoNEU = 0

    try:
        recallPOS = matriz[0][0] / (matriz[0][0] + matriz[0][1] + matriz[2][1] + matriz[0][2] + matriz[1][2])
    except ArithmeticError:
        recallPOS = 0

    try:
        recallNEG = matriz[1][1] / (matriz[1][1] + matriz[1][0] + matriz[2][0] + matriz[0][2] + matriz[1][2])
    except ArithmeticError:
        recallNEG = 0

    try:
        recallNEU = matriz[2][2] / (matriz[2][2] + matriz[1][0] + matriz[2][0] + matriz[0][1] + matriz[2][1])
    except ArithmeticError:
        recallNEU = 0

    try:
        f_measurePOS = 2 * ((precisaoPOS * recallPOS) / (precisaoPOS + recallPOS))
    except ArithmeticError:
        f_measurePOS = 0

    try:
        f_measureNEG = 2 * ((precisaoNEG * recallNEG) / (precisaoNEG + recallNEG))
    except ArithmeticError:
        f_measureNEG = 0

    try:
        f_measureNEU = 2 * ((precisaoNEU * recallNEU) / (precisaoNEU + recallNEU))
    except ArithmeticError:
        f_measureNEU = 0

    f_measure = (f_measurePOS + f_measureNEG + f_measureNEU) / 3

    p = []
    p.append('acuracia: ' + str(acuracia * 100) + '%\n')
    p.append('erro: ' + str(erro * 100) + '%\n')
    p.append('precisao POS: ' + str(precisaoPOS * 100) + '%\n')
    p.append('precisao NEG: ' + str(precisaoNEG * 100) + '%\n')
    p.append('precisao NEUTRO: ' + str(precisaoNEU * 100) + '%\n')
    p.append('recall POS: ' + str(recallPOS * 100) + '%\n')
    p.append('recall NEG: ' + str(recallNEG * 100) + '%\n')
    p.append('recall NEUTRO: ' + str(recallNEU * 100) + '%\n')
    p.append('f-measure: ' + str(f_measure * 100) + '%\n')

    return p, [acuracia, erro, precisaoPOS, precisaoNEG, precisaoNEU, recallPOS, recallNEG, recallNEU, f_measure]


def printMatriz(matriz):
    p = []
    p.append('\tMATRIZ DE CONFUSAO\n')
    p.append('\t\tpos\t| neg | neutro\n')
    p.append('pos\t\t' + str(matriz[0][0]) + '\t| ' + str(matriz[0][1]) + '\t  |\t' + str(matriz[0][2]) + '\n')
    p.append('neg\t\t' + str(matriz[1][0]) + '\t| ' + str(matriz[1][1]) + '\t  |\t' + str(matriz[1][2]) + '\n')
    p.append('neutro\t' + str(matriz[2][0]) + '\t| ' + str(matriz[2][1]) + '\t  |\t' + str(matriz[2][2]) + '\n')

    return p