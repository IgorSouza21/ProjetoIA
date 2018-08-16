import matplotlib.pyplot as plt


def pegarValores(string):
    arq = open('resultados/' + string + '.txt', 'r')
    v = arq.readlines()
    v = v[6].split('acuracia: ')
    v = v[1].split('%\n')
    v = v[0]
    arq.close()

    return v


def pegarValoresCross(string):
    arq = open('resultados/' + string + '.txt', 'r')
    v = arq.readlines()
    v = v[0].split('acuracia: ')
    v = v[1].split('%\n')
    v = v[0]
    arq.close()

    return v


def retornaAcuracias():
    acuraciasBIN = []
    acuraciasBIN.append(round(float(pegarValores('tratamento binário NLTK')), 2))
    acuraciasBIN.append(round(float(pegarValores('tratamento binário SPACY')), 2))

    acuraciasTF = []
    acuraciasTF.append(round(float(pegarValores('tratamento TF NLTK')), 2))
    acuraciasTF.append(round(float(pegarValores('tratamento TF normalizado NLTK')), 2))
    acuraciasTF.append(round(float(pegarValores('tratamento TF SPACY')), 2))
    acuraciasTF.append(round(float(pegarValores('tratamento TF normalizado SPACY')), 2))
    acuraciasTF.append(round(float(pegarValores('Jpretext TF')), 2))
    acuraciasTF.append(round(float(pegarValores('Jpretext Não Normalizado TF')), 2))

    acuraciasTFIDF = []
    acuraciasTFIDF.append(round(float(pegarValores('tratamento TFIDF NLTK')), 2))
    acuraciasTFIDF.append(round(float(pegarValores('tratamento TFIDF normalizado NLTK')), 2))
    acuraciasTFIDF.append(round(float(pegarValores('tratamento TFIDF SPACY')), 2))
    acuraciasTFIDF.append(round(float(pegarValores('tratamento TFIDF normalizado SPACY')), 2))
    acuraciasTFIDF.append(round(float(pegarValores('Jpretext TFIDF')), 2))
    acuraciasTFIDF.append(round(float(pegarValores('Jpretext Não Normalizado TFIDF')), 2))

    return [acuraciasBIN, acuraciasTF, acuraciasTFIDF]


def crossAcuracias():
    acuraciasJpretext = []
    acuraciasJpretext.append(round(float(pegarValoresCross('5-folds com Jpretext TF')), 2))
    acuraciasJpretext.append(round(float(pegarValoresCross('10-folds com Jpretext TF')), 2))
    acuraciasJpretext.append(round(float(pegarValoresCross('15-folds com Jpretext TF')), 2))

    acuraciasNLTK = []
    acuraciasNLTK.append(round(float(pegarValoresCross(u'5-folds com tratamento binário NLTK')), 2))
    acuraciasNLTK.append(round(float(pegarValoresCross(u'10-folds com tratamento binário NLTK')), 2))
    acuraciasNLTK.append(round(float(pegarValoresCross(u'15-folds com tratamento binário NLTK')), 2))

    acuraciasSpacy = []
    acuraciasSpacy.append(round(float(pegarValoresCross(u'5-folds com tratamento binário Spacy')), 2))
    acuraciasSpacy.append(round(float(pegarValoresCross(u'10-folds com tratamento binário Spacy')), 2))
    acuraciasSpacy.append(round(float(pegarValoresCross(u'15-folds com tratamento binário Spacy')), 2))

    acuraciasSem = []
    acuraciasSem.append(round(float(pegarValoresCross('5-folds sem tratamento')), 2))
    acuraciasSem.append(round(float(pegarValoresCross('10-folds sem tratamento')), 2))
    acuraciasSem.append(round(float(pegarValoresCross('15-folds sem tratamento')), 2))

    return [acuraciasJpretext, acuraciasNLTK, acuraciasSpacy, acuraciasSem]


def gerarGraficos(valores):
    plt.title('COMPARANDO AS FERRAMENTAS')
    plt.xlabel('Ferramenta')
    plt.ylabel(u'Acurácia')
    x1 = ['NLTK', 'SPACY']
    x2 = ['NLTK', 'NLTKN', 'SPACY', 'SPACYN', 'JpretextN', 'Jpretext']
    binario = plt.plot(x1, valores[0], 'r-', label=u'binário')
    tf = plt.plot(x2, valores[1], 'b-', label='TF')
    tfidf = plt.plot(x2, valores[2], 'g-', label='TFIDF')
    plt.xticks(x2, [i for i in x2], rotation=45)
    plt.legend()
    plt.show()


def geraGraficoCross(valores):
    plt.title('COMPARANDO AS FERRAMENTAS')
    plt.xlabel('Folds')
    plt.ylabel(u'Acurácia')
    x1 = ['5', '10', '15']
    jpretext = plt.plot(x1, valores[0], 'r-', label=u'Jpretext')
    nltk = plt.plot(x1, valores[1], 'b-', label=u'Binário NLTK')
    spacy = plt.plot(x1, valores[2], 'g-', label=u'Binário Spacy')
    sem = plt.plot(x1, valores[3], 'k-', label='Sem tratamento')
    plt.xticks(x1, [i for i in x1])
    plt.legend()
    plt.show()


def graficoBarra():
    acuracia = []
    acuracia.append(round(float(pegarValores('resultados com Jpretext TF')), 2))
    acuracia.append(round(float(pegarValores('resultados com tratamento binário NLTK')), 2))
    acuracia.append(round(float(pegarValores('resultados com tratamento binário Spacy')), 2))
    acuracia.append(round(float(pegarValores('resultados sem tratamento')), 2))

    plt.title('Melhores Resultados')
    plt.ylabel(u'Acurácia')
    plt.xlabel('Ferramentas')
    plt.bar(['Jpretext TF', 'Binário NLTK', 'Binário Spacy', 'Sem Tratamento'], acuracia,
            color=['yellow', 'blue', 'red', 'green'], align='center')
    plt.show()