forward-synthesis prediction transition-metal catalized reactions
====
#Explain:
#The traditional chemical reaction prediction method uses the 18 electron rule. This research is to predict whether the transfer metal catalyst reaction takes place by using the combination of quantum 

#According to Schrodinger wave equation, s, p, d of atomic hybrid orbit are calculated to make hamilton matrix. After diagonalization, dimension reduction, feature extraction, training in GNN network, 

#experiment：
##step1.After making hamilton matrix c, torch. finally. eigh (c), take the middle 10 values and the original data to do the least square method. Calculate loss to verify the accuracy of the matrix.
    1.1  test.py ：Extract data from the. out file and write it to the data.txt file
    1.2  Read data from data.txt file, generate matrix, control 23 learning parameters, and calculate loss.
    Test results: We arrange the molecular loss from large to small, and check the corresponding molecular structure. We can find that the molecules with large loss have F, CL, Br, I and the smaller ones are N, O and other functional groups.
    ![image](https://github.com/sudehong/forward-synthesis-prediction-transition-metal-catalized-reactions/tree/master/step1/head5.jpg)
    
##step2: Still in process

##step3:
