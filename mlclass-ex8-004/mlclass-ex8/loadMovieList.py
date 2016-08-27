def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    #cell array of the words
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
    #   and returns a cell array of the words in movieList.


    ## Read the fixed movieulary list
    with open("movie_ids.txt") as movie_ids_file:

        # Store all movies in movie list
        n = 1682  # Total number of movies 

        movieList = [None]*n
        for i, line in enumerate(movie_ids_file.readlines()):
            movieName = line.split()[1:]
            movieList[i] = " ".join(movieName)

    return movieList