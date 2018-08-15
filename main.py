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
    tratBinMultNLTK(textos, polaridades, classes, biblioteca1)

    tratBinSpacy(textos, polaridades, classes, biblioteca2)
    tratBinMultSpacy(textos, polaridades, classes, biblioteca2)

    tratTFNLTK(textos, polaridades, classes, biblioteca1)
    tratTFnormalizadoNLTK(textos, polaridades, classes, biblioteca1)
    tratTFMultNLTK(textos, polaridades, classes, biblioteca1)
    tratTFnormalizadoMultNLTK(textos, polaridades, classes, biblioteca1)

    tratTFSpacy(textos, polaridades, classes, biblioteca2)
    tratTFNormalizadoSpacy(textos, polaridades, classes, biblioteca2)
    tratTFMultSpacy(textos, polaridades, classes, biblioteca2)
    tratTFNormalizadoMultSpacy(textos, polaridades, classes, biblioteca2)

    tratTFIDFNLTK(textos, polaridades, classes, biblioteca1)
    tratTFIDFnormalizadoNLTK(textos, polaridades, classes, biblioteca1)
    tratTFIDFMultNLTK(textos, polaridades, classes, biblioteca1)
    tratTFIDFnormalizadoMultNLTK(textos, polaridades, classes, biblioteca1)

    tratTFIDFSpacy(textos, polaridades, classes, biblioteca2)
    tratTFIDFNormalizadoSpacy(textos, polaridades, classes, biblioteca2)
    tratTFIDFMultSpacy(textos, polaridades, classes, biblioteca2)
    tratTFIDFNormalizadoMultSpacy(textos, polaridades, classes, biblioteca2)

    tratTFJpretext(classes)
    tratTFNaoNormalizadoJpretext(classes)
    tratTFMultJpretext(classes)
    tratTFNaoNormalizadoMultJpretext(classes)

    tratTFIDFJpretext(classes)
    tratTFIDFNaoNormalizadoJpretext(classes)
    tratTFIDFMultJpretext(classes)
    tratTFIDFNaoNormalizadoMultJpretext(classes)


def testeK_folds(k, textos, polaridades, classes, biblioteca1, biblioteca2):
    pass


main()

