# LT2212 V20 Assignment 1

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

Part 1: I chose to load in the contents of the file using the nltk tokeniser and keeping all non letters in, as well as not lower casing.

Part 4: In the plot of the tfidf scores you can see that some of the medium frequent words get pushed to the top. These are the words that are way more frequent in documents of one class vs the other (e.g. oil and tonnes). 
You can also see some extremely frequent words and punctuation marks in the top scores. I assume this is due to the very high raw counts, that elevate the presumably low idf scores a lot. It would make sense to filter these out to get more value out of the tfidf.

Part Bonus: I chose to run the KNeighborsClassifier on the data, setting the n_neighbours to 10. This way I got an accuracy score of0.8060344827586207 for the raw counts and 0.8706896551724138 when run on the tfidf scores. Ideally tfidf should focus on the words that are more frequent in one class (clearly showing the class of a document) and therefore raise the score of the classifier. 
Since my tfidf included so many non content words, it didn't actually make that big of a but still a noticable difference.
Weirdly enough this only worked when I ran it in my terminal, not on the Jupiter Notebook.
