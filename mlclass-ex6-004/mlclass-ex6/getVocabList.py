def getVocabList():
    #GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    #cell array of the words
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
    #   and returns a cell array of the words in vocabList.


    ## Read the fixed vocabulary list
    with open('vocab.txt', 'r') as vocabFile:

        # Store all dictionary words in dictionary vocabList
        vocabList = {}
        for line in vocabFile.readlines():
            i, word = line.split()
            vocabList[word] = int(i)

    return vocabList
