#!/usr/bin/python
import pickle
import itertools
import sys
import math
from operator import itemgetter
from nltk.corpus import brown # using the brown corpus

DICT = {}

def categorize(text):
    counts = {}
    string = ' '.join(brown.words(text))
    counts = generate_ngrams(string, counts)
    cosine_estimate = "NO CATEGORY"
    cosine_min = sys.maxint
    out_of_place_estimate = "NO CATEGORY"
    out_of_place_min = sys.maxint
    for category in DICT:
        tcos = cosine_measure(DICT[category], counts)
        if tcos < cosine_min:
            cosine_estimate = category
            cosine_min = tcos
        tout = out_of_place_measure(DICT[category], counts)
        if tout < out_of_place_min:
            out_of_place_estimate = category
            out_of_place_min = tout
    return (cosine_estimate, out_of_place_estimate)

def cosine_measure(template, sample):
    """ cosine_measure: compares the difference between template
        and sample dictionaries by comparing the cosine difference """
     # fill sample with all items not in template
    for key in template not in sample:
        sample[key] = 0
    tvector = sorted(template.iteritems(), key=itemgetter(1), reverse=True)
    svector = []
    # order svector to be the same ngram ordering as tvector
    for tup in tvector:
        svector.append(sample[tup[0]])
    # turn absolute values into frequencies
    ttotal = sum([tup[1] for tup in tvector])
    stotal = sum([tup[1] for tup in svector])
    tfreqs = [tup[1]/ttotal for tup in tvector]
    sfreqs = [tup[1]/stotal for tup in svector]
    return cosine_dist(tfreqs, sfreqs)

def cosine_dist(vect1, vect2):
    mu1 = 1/len(vect1) * sum(vect1)
    mu2 = 1/len(vect2) * sum(vect2)
    times = [(x - mu1) * (y - mu2) for x in vect1 for y in vect2]
    sqr1 = [x ** 2 - mu1 for x in vect1]
    sqr2 = [x ** 2 - mu2 for x in vect2]
    return math.acos(sum(times)/math.sqrt(sum(sqr1)*sum(sqr2)))


def out_of_place_measure(template, sample):
    """ out_of_place_measure: compares the difference between template
        and sample by means of couting the number of inversions. """
    tvector = sorted(template.iteritems(), key=itemgetter(1), reverse=True)
    svector = sorted(sample.iteritems(), key=itemgetter(1), reverse=True)
    svector_ordering = []
    for i in xrange(0, len(tvector)):
        if tvector[i] in svector:
            index = svector.index(svector[i])
            svector_ordering.append(index)
        else:
            # TODO: What if ngram is not present in svector
            svector_ordering.append(len(tvector))
    return MergeCount(svector_ordering)[0]

def MergeCount(A):
    """ returns number of inversions and sorted array """
    if len(A) < 2:
        return (0, A)
    mid = len(A) / 2
    return Merge(MergeCount(A[:mid]), MergeCount(A[mid:]))

def Merge(aTuple, bTuple):
    inversions = aTuple[0] + bTuple[0]
    l = aTuple[0]
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


def generate():
    for category in brown.categories():
        DICT[category] = generate_corpus(brown.fileids(categories=category))
    dict_file = open('ngrams.pk1', 'wb')
    pickle.dump(DICT, dict_file)
    dict_file.close()
    print DICT


# where corpus is the 'news' category or something like that
def generate_corpus(corpus):
    counts = {}
    for text in corpus:
        string = ' '.join(brown.words(text))
        generate_ngrams(string, counts)
    return counts

def generate_ngrams(string, counts):
    lower = 3
    upper = 6
    my_range = range(lower, upper)
    my_strings = []
    # generate all strings
    for i in range(0, upper):
        my_strings.append(string[i:])
    # generate all N-grams (in myrange) for this string
    for i in my_range:
        strings = my_strings[0:i]
        for gram in itertools.izip(*strings):
            if gram in counts:
                counts[gram] += 1
            else:
                counts[gram] = 1
    return counts
        

if __name__ == "__main__":
    generate()
