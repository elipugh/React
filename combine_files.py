import os
import csv
import re
import nltk
import string
import pandas as pd
import numpy as np

def extractCsv():
    folder = 'data/data1'
    # out = open(folder + "_compiled.csv", "w+")
    out = open("data2_compiled.csv", "w+")
    files = os.listdir('./' + folder + "/")

    dataframes = []
    for i in files:
        df = pd.read_csv(folder + "/" + i)
        dataframes.append(df)

    if len(dataframes) == 0:
        print("No files available.")
        return

    df = pd.concat(dataframes, sort=True)
    df = df.reset_index(drop=True)
    df.assign(
        # lowercase, removed repeated sequences, removed punc
        clean_status_message = '',
        # has repeats >= 3 characters (i. e. 'abbbb')
        has_long_repeats = 0,
    )

    # rote features
    df['has_exclamation'] = df.apply(lambda row: hasChar('!', row), axis=1)
    df['has_question_mark'] = df.apply(lambda row: hasChar('?', row), axis=1)
    df['has_link'] = df.apply(lambda row: hasLink(row), axis=1)

    for index, row in df.iterrows():
        if isinstance(row['status_message'], str):
            processRow(df, index)
        else:
            df.drop([index])

    df.to_csv(out)
    out.close()

def processRow(df, index):
    cleanStatus, hasLongRepeats = removeLongRepeats(df.at[index, 'status_message'])
    cleanStatus = getWithoutLinks(cleanStatus)
    cleanStatus = getWithoutPunctuation(cleanStatus)
    cleanStatus = cleanStatus.lower()

    df.at[index, 'clean_status_message'] = cleanStatus
    df.at[index, 'has_long_repeats'] = hasLongRepeats

    return

def removeLongRepeats(str):
    curr = ""
    currCount = 0
    cleanedstr = ""
    hasLongRepeats = False

    for char in str:
        if curr == char:
            currCount += 1
        else:
            curr = char
            currCount = 1

        if currCount <= 3:
            cleanedstr += char
        else:
            hasLongRepeats = True

    return cleanedstr, hasLongRepeats

def hasChar(char, row):
    if isinstance(row['status_message'], str):
        return 1 if (char in row['status_message']) else 0
    return 0

def hasLink(row):
    if isinstance(row['status_message'], str):
        return 1 if re.search(r'http\S+', row['status_message']) else 0
        # return 1 if re.search(r'^https?:\/\/.*[\r\n]*', row['status_message']) else 0
    return 0

def getWithoutLinks(str):
    return re.sub(r'http\S+', '', str)

def getWithoutPunctuation(str):
    return str.translate(str.maketrans('', '', string.punctuation))

# removeUnnecessaryPunctuation("jjjj\"''sa\"jl")
writeCsv()
