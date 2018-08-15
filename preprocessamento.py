import math
import operator


def document(t, c):
    return [t,c]


def dicionario(ts, k):
    hs = {}

    for tok in ts:
        if tok in hs:
            hs[tok] += 1
        else:
            hs[tok] = 1

    sot = sorted(hs.items(), key=operator.itemgetter(1))
    sk = []
    df = []
    for i in sot:
        sk.append(i[0])
        df.append(i[1])

    fs = sk[k:]
    df = df[k:]
    return fs, df


def separaPositivosNegativosNeutros(textos, polaridades):
    neg = []
    pos = []
    neutro = []
    for i in range(len(textos)):
        if polaridades[i] is 0:
            neutro.append(document(textos[i], polaridades[i]))
        elif polaridades[i] is 1:
            pos.append(document(textos[i], polaridades[i]))
        elif polaridades[i] is -1:
            neg.append(document(textos[i], polaridades[i]))
    l = [neutro, pos, neg]

    return sorted(l)


def estratificacao(lista, total):
    tamanhos = []
    porcentagens = []
    for l in lista:
        tamanhos.append(len(l))

    for tamanho in tamanhos:
        porcentagens.append(tamanho/total)

    return porcentagens

