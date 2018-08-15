import os
import lerXmlInputProjeto
from TratamentosNaiveBayes import *


def main():
    biblioteca1 = Type.NLTK
    biblioteca2 = Type.SPACY
    classes = [0, 1, -1]
    textos = lerXmlInputProjeto.selecionarTodosTextos()
    polaridades = lerXmlInputProjeto.selecionarPolaridades()

    if pasta is not '':
        if not os.path.exists(pasta):
            os.mkdir(pasta)

    testesHoldout(textos, polaridades, classes, biblioteca1, biblioteca2)


def testesHoldout(textos, polaridades, classes, biblioteca1, biblioteca2):

    semTratamento(textos, polaridades, classes)

    tratBinNLTK(textos, polaridades, classes, biblioteca1)

    tratBinSpacy(textos, polaridades, classes, biblioteca2)

    tratTFNLTK(textos, polaridades, classes, biblioteca1)
    tratTFnormalizadoNLTK(textos, polaridades, classes, biblioteca1)

    tratTFSpacy(textos, polaridades, classes, biblioteca2)
    tratTFNormalizadoSpacy(textos, polaridades, classes, biblioteca2)

    tratTFIDFNLTK(textos, polaridades, classes, biblioteca1)
    tratTFIDFnormalizadoNLTK(textos, polaridades, classes, biblioteca1)

    tratTFIDFSpacy(textos, polaridades, classes, biblioteca2)
    tratTFIDFNormalizadoSpacy(textos, polaridades, classes, biblioteca2)

    tratTFJpretext(classes)
    tratTFNaoNormalizadoJpretext(classes)

    tratTFIDFJpretext(classes)
    tratTFIDFNaoNormalizadoJpretext(classes)


def testeK_folds(k, textos, polaridades, classes):
    r = crossValidation(k, textos, polaridades, classes, Type.NLTK, Type.BIN)
    s = 'acuracia: '
    arq = open(pasta + 'resultados com tratamento binário NLTK.txt', 'w')
    arq.writelines(s1)
    arq.write('\n')
    arq.writelines(s2)
    arq.close()


main()

