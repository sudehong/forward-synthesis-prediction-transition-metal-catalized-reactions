import os
import os.path
import numpy as np
import re

def string_to_float(str):
    return float(str)

def forwardToeigenvalues(i,l1):
    for k in range(i,len(l1) , 1):
        if (l1[k].strip() == ''):
            continue

        l8 = l1[k - 1].split()
        l6 = l1[k].split()
        l7 = l1[k+1].split()

        virt = []
        occ = []

        if l6[0] == 'Alpha':
            if l7[1] == 'virt.':
                #提取特征值，提取virt.
                virt = [i for i in l7[4:]]
                #提取occ
                add = []
                s =   [i for i in l8[4:]] + [i for i in l6[4:]]
                add = s

                #取后面五个
                occ = add[-5:]
                return occ + virt
        else:
            continue






def find_small_energy(path):
    #read the file
    with open(path) as f:
        l1 = f.readlines()
    #find the smaller eneger and force in each line


    for i in range(len(l1)):

        if (l1[i].strip() == 'Item               Value     Threshold  Converged?'):
            l2 = l1[i + 1].split()
            l3 = l1[i + 2].split()
            l4 = l1[i + 3].split()
            l5 = l1[i + 4].split()
            if l2[4] == 'YES' and l3[4] == 'YES' and l4[4] == 'YES' and l5[4] == 'YES':
                l = forwardToeigenvalues(i,l1)
                return l




file_dir = 'C:\\Users\\SU DEHONG\\Desktop\\project\\molecule_hamilton_matrix'
e = {}
for root, dirs, files in os.walk(file_dir):
    # print(root)
    # print(files)
    for file in files:

        if file.endswith('.out'):
            path = os.path.join(root, file)
            eigenvalues = find_small_energy(path)
            #提取文件名数字
            regex = re.compile(r'\d+')
            e.update({str(max(regex.findall(file))):eigenvalues})

#匹配txt，找出smile
def find_smile(key):
    f = open("JMedChem.34.786-smiles.txt",'r')
    lines = f.readlines()
    for line in lines:
        l = line.split()
        if  int(l[0]) == int(key):
            return l[2]


# 写之前，先检验文件是否存在，存在就删掉
if os.path.exists("data.txt"):
    os.remove("data.txt")

file_write_obj = open("data.txt", 'w')
for (key,value) in e.items():
    smile = find_smile(key)
    file_write_obj.writelines(key+'  ')
    file_write_obj.writelines('  '.join(value) )
    file_write_obj.writelines('  ')
    file_write_obj.writelines(str(smile))

    file_write_obj.write('\n')
file_write_obj.close()
















