#Exploratory Data Analysis.
#Please check the required set of libraries are installed. 
#all the references are mentioned in the code.

import spacy
from textstat.textstat import textstatistics, legacy_round
import os
import nltk
from nltk import *
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from subprocess import Popen, STDOUT, PIPE

#reading score calculation.
#the code is taken from another source, which is mentioned below.
def break_sentences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc.sents

def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words

def avg_word_length(text):
    words = word_count(text)
    sentences = 1
    average_word_length = float((len(text) + 1 - words)/ words)
    return average_word_length

def syllables_count(word):
    return textstatistics().syllable_count(word)

def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)

def readability_score(text):
    read_sc = 206.835 - float(1.015 * avg_word_length(text)) -\
          float(84.6 * avg_syllables_per_word(text))

    return legacy_round(read_sc, 2)

#postag-StanfordPOSTagger
#stanford tutorial which is mentioned in the references

java_path = "/Library/Java/JavaVirtualMachines/jdk-16.0.1.jdk/Contents/Home/bin/java"
os.environ["JAVAHOME"] = java_path

jar = "/Users/oldxchange/Downloads/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
model = "/Users/oldxchange/Downloads/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger"

def posVector(tagged_words):
    vector = list()
    for tag in ['NNP', 'DT', 'IN', 'JJ','NN', 'NNS']:
        temp = 0
        for tup in tagged_words:
            if(tup[1] == tag and temp==0):
                vector.append(1)
                temp = 1
            else:
                continue
        if temp==1:
            continue
        else:
            vector.append(0)

    return vector

def pos_tagging(text):
    #defining global variable
    #download latest version of: https://nlp.stanford.edu/software/tagger.shtml
    global jar
    global model

    pos_tagger = StanfordPOSTagger(model, jar, encoding = "utf-8")
    words = nltk.word_tokenize(text)
    tagged_words = pos_tagger.tag(words)

    return tagged_words

#Clickbait detector

def clickbait(text):
    #have used the code from the githubrepo: https://github.com/saurabhmathur96/clickbait-detector
    handle = Popen("python <clickbaitlocation>/src/detect.py " + str(text) ,shell=True, stdout=PIPE, stderr=STDOUT, stdin=PIPE)
    output = handle.stdout.read()
    temp = check.split(b"headline is ")[1].split(b" % ")[0]

    return float(temp)
