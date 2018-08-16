import os
import lerXmlInputProjeto
from TratamentosNaiveBayes import *
from crossNaiveBayes import *
import timeit
import graficos as g


def main():
    biblioteca1 = Type.NLTK
    biblioteca2 = Type.SPACY
    classes = [0, 1, -1]
    textos = lerXmlInputProjeto.selecionarTodosTextos()
    polaridades = lerXmlInputProjeto.selecionarPolaridades()

    if pasta is not '':
        if not os.path.exists(pasta):
            os.mkdir(pasta)

    # testesHoldout(textos, polaridades, classes, biblioteca1, biblioteca2)
    testeK_folds(15, textos, polaridades, classes, biblioteca1, biblioteca2)


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


def testeK_folds(k, textos, polaridades, classes, biblioteca1, biblioteca2):
    inicio = timeit.default_timer()
    arq = crossBinNLTK(k, textos, polaridades, classes, biblioteca1)
    fim = timeit.default_timer()
    arq.write('\ntempo de execucao: ' + str(fim - inicio))
    arq.close()

    inicio = timeit.default_timer()
    arq = crossTFnormalizadoJpretext(k, classes)
    fim = timeit.default_timer()
    arq.write('\ntempo de execucao: ' + str(fim - inicio))
    arq.close()

    inicio = timeit.default_timer()
    arq = crossBinSpacy(k, textos, polaridades, classes, biblioteca2)
    fim = timeit.default_timer()
    arq.write('\ntempo de execucao: ' + str(fim - inicio))
    arq.close()

    inicio = timeit.default_timer()
    arq = crossSemTratamento(k, textos, polaridades, classes)
    fim = timeit.default_timer()
    arq.write('\ntempo de execucao: ' + str(fim - inicio))
    arq.close()

# g.gerarGraficos(retornaAcuracias())
# g.geraGraficoCross(crossAcuracias())
# g.graficoBarra()
# main()


