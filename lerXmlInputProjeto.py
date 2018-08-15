import os
from xml.etree import ElementTree

file_name = 'inputProjeto.xml'
full_file = os.path.abspath(os.path.join('data', file_name ))
tree = ElementTree.parse(file_name)
root = tree.getroot()


def selecionarTodosTextos():
    dom = ElementTree.parse(file_name)
    todosTextos = dom.findall('Review/sentences/sentence/text')
    vTextos = []
    for i in range(0, len(todosTextos)):
        vTextos.append(todosTextos[i].text)
    return vTextos


def polaridade(polar):
    neg = 0
    pos = 0
    for i in range(0, len(polar)):
        if polar[i] == 'negative':
            neg += 1
        elif polar[i] == 'positive':
            pos += 1
    if pos > neg:
        return 1
    elif pos < neg:
        return -1
    else:
        return 0


def selecionarPolaridades():
    polar = []
    for i in range(0,len(root)): #For iterando a variável i de  0 até o tamanho do root
        j = 0
        while j < (len(root[i][0])): #While para verificar as 'sentece'
            vPolaridade = []
            for k in range(0, len(root[i][0][j][1])):
                vPolaridade.append(root[i][0][j][1][k].get('polarity'))
                polar.append(polaridade(vPolaridade))
            j += 1
    return polar
