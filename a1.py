import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import nltk
import numpy as np
import numpy.random as npr
from glob import glob
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def part1_load(folder1, folder2, n=1):
    #load documents
    class1 = glob("{}/*.txt".format(folder1))
    class2 = glob("{}/*.txt".format(folder2))
    big_table = {folder1: {}, folder2: {}}
    all_words = []
    listoffiles1 = []
    listoffiles2 = []
    df = pd.DataFrame()
    # get content of documents in folder 1
    for filename in class1:
        listoffiles1.append(filename)
        filecontent = ""
        with open(filename, "r") as thefile:
            for line in thefile:
                filecontent += line
            tokenized = nltk.word_tokenize(filecontent)
            big_table[folder1][filename] = {}
            # count words in each file
            for word in tokenized:
                if word not in big_table[folder1][filename]:
                    big_table[folder1][filename][word] = 1
                else:
                    big_table[folder1][filename][word] += 1
                if word not in all_words:
                    all_words.append(word)
    # get content of documents in folder 2
    for filename in class2:
        listoffiles2.append(filename)
        filecontent = ""
        with open(filename, "r") as thefile:
            for line in thefile:
                filecontent += line
            tokenized = nltk.word_tokenize(filecontent)
            big_table[folder2][filename] = {}
            # count words in each file
            for word in tokenized:
                if word not in big_table[folder2][filename]:
                    big_table[folder2][filename][word] = 1
                else:
                    big_table[folder2][filename][word] += 1
                if word not in all_words:
                    all_words.append(word)
    # fill in words that are not in the file
    for filename in class1:
        for word in all_words:
            if word not in big_table[folder1][filename]:
                big_table[folder1][filename][word] = 0
    for filename in class2:
        for word in all_words:
            if word not in big_table[folder2][filename]:
                big_table[folder2][filename][word] = 0
    pdict = {}
    wordlist = []
    # sort wordcounts and compare to input value n
    for word in all_words:
        plist = []
        for filename in listoffiles1:
            plist.append(big_table[folder1][filename][word])
        for filename in listoffiles2:
            plist.append(big_table[folder2][filename][word])
        if sum(plist) > n:
            wordlist.append(word)
            pdict[word] = plist
    listofallfiles = listoffiles1 + listoffiles2
    listoffolder1 = [folder1 for i in range(len(listoffiles1))]
    listoffolder2 = [folder2 for j in range(len(listoffiles2))]
    listoffolders = listoffolder1 + listoffolder2
    # construct dataframe
    df['filename'] = listofallfiles
    df['classname'] = listoffolders
    for word in wordlist:
        df[word] = pdict[word]
    return df

def part2_vis(df, m):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    wordsx = list(df)
    words = wordsx[2:]
    count_dict = {}
    for word in words:
        amount = int(df.loc[:, word].sum())
        count_dict[word] = amount
    sorted_counts = sorted(count_dict, key=count_dict.get, reverse=True)
    to_plot = sorted_counts[:m]
    sorted_by_folder = df.groupby(by=['classname']).sum()
    final_df = pd.DataFrame()
    for word in to_plot:
        final_df[word] = sorted_by_folder[word]
    plotting = final_df.T.plot.bar()
    return plotting

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    classnames = df['classname'].copy()
    del df['classname']
    filenames = df['filename'].copy()
    del df['filename']
    number_docs = df.shape[0]
    not_zero = df.astype(bool).sum(axis=0)
    idf = np.log((number_docs/not_zero))
    transformed_df = df.mul(idf, axis=1)
    transformed_df.insert(0, 'classname', classnames)
    transformed_df.insert(0, 'filename', filenames)
    return transformed_df

def classify(df):
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    y = df['classname'].values
    X = df.iloc[:,2:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    KNN_model.fit(X_train, y_train)
    KNN_prediction = KNN_model.predict(X_test)
    score = accuracy_score(KNN_prediction, y_test)
    return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Classifies the input documents.")
    parser.add_argument("folder1", type=str, help="Name of the first directory.")
    parser.add_argument("folder2", type=str, help="Name of the second directory.")
    parser.add_argument("n", type=int, nargs='?', default=1, help="Minimum count.")

    args = parser.parse_args()

    print(classify(part1_load(args.folder1, args.folder2, args.n)))

