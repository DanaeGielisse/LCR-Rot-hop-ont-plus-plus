with open('data/programGeneratedData/allEmbeddings.txt','w') as outf:
    with open('data/temporaryData/testEmbeddings.txt', 'r') as out1:
        list_1 = out1.readlines()
        for i in range(0,len(list_1)):
            outf.write(list_1[i])
    with open('data/temporaryData/testEmbeddings1.txt', 'r') as out2:
        list_2 = out2.readlines()
        for i in range(0,len(list_2)):
            outf.write(list_2[i])
    with open('data/temporaryData/testEmbeddings2.txt', 'r') as out3:
        list_3 = out3.readlines()
        for i in range(0,len(list_3)):
            outf.write(list_3[i])
    with open('data/temporaryData/testEmbeddings3.txt', 'r') as out4:
        list_4 = out4.readlines()
        for i in range(0,len(list_4)):
            outf.write(list_4[i])
    with open('data/temporaryData/testEmbeddings4.txt', 'r') as out5:
        list_5 = out5.readlines()
        for i in range(0,len(list_5)):
            outf.write(list_5[i])
