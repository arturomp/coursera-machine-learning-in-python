def readFile(filename):
    #READFILE reads a file and returns its entire contents 
    #   file_contents = READFILE(filename) reads a file and returns its entire
    #   contents in file_contents
    #

    # Load File
    try:
        with open(filename, 'r') as openFile:
            file_contents = openFile.read()
    except:
        file_contents = ''
        print('Unable to open {:s}'.format(filename))

    return file_contents
