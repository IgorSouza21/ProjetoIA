import xml.etree.cElementTree as ET
tree = ET.parse('ABSA-15_Restaurants_Train_Final.xml')
root = tree.getroot()

for i in range(0,len(root)):
    j = 0
    while j < (len(root[i][0])):
        try:
           root[i][0][j][1]
        except IndexError:
            root[i][0].remove(root[i][0][j])
            j -= 1
        j += 1
tree.write('inputProjeto.xml', encoding='UTF-8')
