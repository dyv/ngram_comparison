#!/usr/bin/python
from __future__ import division
from bisect import bisect_left
import pickle
import itertools
import string
import sys
import math
import re
from operator import itemgetter
from nltk.corpus import brown # using the brown corpus

DICT = {}
LARGEST_NGRAM = 6
SMALLEST_NGRAM = 1
TRAINING_SET = 10
COMPARISON_SET = 5

def categorize_all():
    cos_correct = 0
    oop_correct = 0
    total = 0
    count = 0
    max_count = 5
    cats = brown.categories()
    for category in cats:
        for text in brown.fileids(category)[TRAINING_SET:]:
            print "CATEGORIZING: " + text
            print "EXPECTED CATEGORY: " + category
            if count >= COMPARISON_SET:
                count = 0
                break
            count += 1
            total += 1
            cos, oop = categorize(text)
            if cos == category:
                print ("CORRECT: Cosine Estimate of " + text + ": Actual = " + cos + ", Expected = " + category)
                cos_correct += 1
            else:
                print "INCORRECT: Cosine Estimate of " + text + ": Actual = " + cos + ", Expected = " + category

            if oop == category:
                print "CORRECT: Out of Place of " + text + ": Actual = " + oop  + ", Expected = " + category
                oop_correct += 1
            else:
                print "INCORRECT: Out of Place of " + text + ": Actual = " + oop + ", Expected = " + category

    print "CORRECT COSINES: " + str(cos_correct)
    print "CORRECT OUT OF PLACE: " + str(oop_correct)
    print "OUT OF: " + str(total)

def categorize(text):
    counts = {}
    counts = generate_text(text, counts)
    cosine_estimate = "NO CATEGORY"
    cosine_min = sys.maxint
    out_of_place_estimate = "NO CATEGORY"
    out_of_place_min = sys.maxint
    #print "Beginning Catigorizations For " + text
    for category in DICT:
        if category == "science_fiction" or category == "humor":
            continue
        #print "CATEGORY IS: " + category
        tcos = cosine_measure(DICT[category], counts)
        #print category + ": tcos = "
        #print tcos
        if tcos < cosine_min:
        #    print "tcos better than minsofar: " + category
        #    #print "dist = "
        #    #print tcos
            cosine_estimate = category
            cosine_min = tcos
        tout = out_of_place_measure(DICT[category], counts)
        #print category + ": inversions = " + str(tout)
        if tout < out_of_place_min:
            #print "tout better than minsofar: " + category
            out_of_place_estimate = category
            out_of_place_min = tout
    return (cosine_estimate, out_of_place_estimate)

def cosine_measure(template, sample):
    """ cosine_measure: compares the difference between template
        and sample dictionaries by comparing the cosine difference """
    # fill sample with all items not in template
    for key in template:
        if key not in sample:
            sample[key] = 0

    for key in sample:
        if key not in template:
            template[key] = 0

    tvector = sorted(template.iteritems(), key=itemgetter(0), reverse=True)
    svector = sorted(sample.iteritems(), key=itemgetter(0), reverse=True)

    # turn absolute values into frequencies
    ttotal = sum([tup[1] for tup in tvector])
    stotal = sum([tup[1] for tup in svector])
    tfreqs = [tup[1]/ttotal for tup in tvector]
    sfreqs = [tup[1]/stotal for tup in svector]
    return cosine_dist(tfreqs, sfreqs)

def cosine_dist(vect1, vect2):
    mu1 = 1/len(vect1) * sum(vect1)
    mu2 = 1/len(vect2) * sum(vect2)
    numerator = 0
    sumsqr1 = 0
    sumsqr2 = 0
    for i in xrange(0,len(vect1)):
        numerator += vect1[i]-mu1 * vect2[i]-mu2
        sumsqr1 += vect1[i] ** 2 - mu1
        sumsqr2 += vect2[i] ** 2 - mu2
    dist =  math.acos(numerator/math.sqrt(sumsqr1*sumsqr2))
    return dist 

def search(l, value):
    #print "searching for: " + value
    for i, x in enumerate(l):
        if x[0] == value:
            return i
    return -1

def binary_search(a, x):   # can't use a to specify default for hi
    low = 0
    high = len(a)-1
    while low <= high:
        mid = (low+high) // 2
        if a[mid][0] > x:
            high = mid-1
        elif a[mid][0] < x:
            low = mid+1
        else:
            return mid
    return -1


def out_of_place_measure(template, sample):
    """ out_of_place_measure: compares the difference between template
        and sample by means of couting the number of inversions. """
    tvector = sorted(template.iteritems(), key=itemgetter(1), reverse=True)
    tvector = tvector[300:]
    keyindex = {key[0]: index for index, key in enumerate(tvector)}
    svector = sorted(sample.iteritems(), key=itemgetter(1), reverse=True)
    svector = svector[300:]
    dist = 0
    for i in xrange(0, len(svector)):
        #print svector[i]
        if svector[i][0] in keyindex:
            index = keyindex[svector[i][0]]
            dist += abs(index - i)
            #print "sample in template"
        else:
            # TODO: What if ngram is not present in svector
            dist += abs(len(tvector))
    return dist

def MergeCount(A):
    """ returns number of inversions and sorted array """
    if len(A) < 2:
        return (0, A)
    mid = len(A) // 2
    return Merge(MergeCount(A[:mid]), MergeCount(A[mid:]))

def Merge(aTuple, bTuple):
    inversions = aTuple[0] + bTuple[0]
    l = aTuple[1]
    r = bTuple[1]
    result = []
    i = j = 0
    while i < len(l) and j < len(r):
        if l[0] < r[0]:
            result.append(l[i])
            i += 1
        else:
            result.append(r[j])
            inversions += (len(l) -i)
            j += 1
    result.extend(l[i:])
    result.extend(r[j:])
    return (inversions, result)

def opendict():
    global DICT
    DICT = pickle.load(open('ngrams.pk1', 'rb'))

def generate():
    global DICT
    for category in brown.categories():
        # skip the two categories with too little information
        if category == "science_fiction" or category == "humor":
            continue
        DICT[category] = generate_corpus(brown.fileids(categories=category))
    dict_file = open('ngrams.pk1', 'wb')
    pickle.dump(DICT, dict_file)
    dict_file.close()

    # sort each dictionary in DICT
    #vdict = [sorted(DICT[d].iteritems(), key=itemgetter(1), reverse=True) for d in DICT]
    #log_file = open('dict.txt', 'w')
    #print >>log_file, vdict
    #log_file.close()
    #print "vdict"
    #print vdict

PUNCT_STRING = r"(["+string.punctuation+"])+$"
PUNCT = re.compile(PUNCT_STRING)
# where corpus is the 'news' category or something like that
def generate_corpus(corpus):
    counts = {}
    max_training_items = 0
    for text in corpus:
        if max_training_items >= TRAINING_SET:
            break
        max_training_items += 1
        counts = generate_text(text, counts)
    return counts

def generate_text(text, counts):
    for word in brown.words(text):
        if re.match(PUNCT, word):
            continue
        counts = generate_ngrams(word.lower(), counts)
    return counts


def generate_ngrams(string, counts):
    global SMALLEST_NGRAM, LARGEST_NGRAM
    lower = SMALLEST_NGRAM
    upper = LARGEST_NGRAM
    # pad string with upper-1 spaces on either side
    string = " "*(upper-1) + string + " "*(upper-1)
    my_strings = []
    # generate all strings
    for i in range(0, upper):
        my_strings.append(string[i:])
    # generate all N-grams (in myrange) for this string
    for i in range(lower, upper):
        strings = my_strings[0:i]
        for gram in itertools.izip(*strings):
            gram = "".join(gram)
            if gram.isspace():
                continue
            elif gram in counts:
                counts[gram] += 1
            else:
                counts[gram] = 1
    return counts


if __name__ == "__main__":
    #generate()
    opendict()
    categorize_all()
