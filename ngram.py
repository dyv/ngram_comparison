#!/usr/bin/python
from __future__ import division
from bisect import bisect_left
from multiprocessing import Pool
import copy
import pickle
import itertools
import string
import sys
import math
import re
from operator import itemgetter
from nltk.corpus import brown # using the brown corpus
from nltk.corpus import stopwords

BAD_CATS = ["science_fiction", "humor"]# "belles_lettres", "news", "fiction", "hobbies", "religion"]
CATAGORIES = [x for x in brown.categories() if x not in BAD_CATS]
STOPWORDS = stopwords.words('english')

# DICT: Dictionary of all ngrams [SMALLEST_NGRAM, LARGEST_NGRAM)
# Used in Cavnar's approach
DDICT = {}
CDICT = {}
# LARGEST_NGRAM: Largest size of ngrams to put in DICT non-inclusive
LARGEST_NGRAM = 6
# SMALLEST_NGRAM: Smallest size of ngrams to put in DICT inclusive
SMALLEST_NGRAM = 1
# TRAINING_SET: Maximum number of items (per category) to train on
TRAINING_SET = 5
# COMPARISON_SET: Number of items to check agains
COMPARISON_SET = 5

def categorize_all():
    cos_correct = 0
    oop_correct = 0
    total = 0
    cwrong = []
    owrong = []
    pool = Pool()
    results =  pool.map(categorize_cat, CATAGORIES)
    cats = { x : {} for x in CATAGORIES}
    for res in results:
        # the category itself
        cat = res[0]
        cats[cat] = {}
        # category number classified
        cats[cat]["total"] = res[3]
        # category number correctly classified
        cats[cat]["dright"] = res[1]
        cats[cat]["cright"] = res[2]
        # category number classified incorrectly
        cats[cat]["dwrong"] = res[4]
        #print "DWRONG: " + str(len(cats[cat]["dwrong"]))
        #print "DRIGHT: " + str(res[1])
        # numbers correct at this point
        cats[cat]["cwrong"] = res[5]
        # category accuracy measures
        if res[3] == 0:
            cats[cat]["daccuracy"] = 1
            cats[cat]["caccuracy"] = 1
            cats[cat]["drecall"] = 1
            cats[cat]["crecall"] = 1
        else:
            cats[cat]["daccuracy"] = res[1]/res[3]
            cats[cat]["caccuracy"] = res[2]/res[3]
            # category recall measures recall is just category by category
            # accuracy
            cats[cat]["drecall"] = res[1]/res[3]
            cats[cat]["crecall"] = res[2]/res[3]
        # add to golbal counts
        total += res[3]
        cos_correct += res[1]
        oop_correct += res[2]

    # compute precision
    # true positives vs categorized as belonging to c
    for cat in cats:
        dtotal_classified = cats[cat]["dright"]
        ctotal_classified = cats[cat]["cright"]
        for c in cats:
            for w in cats[c]["dwrong"]:
                if w == "NO CATEGORY":
                    print "IMPROPERLY CLASSIFIED DAMASHEK: " + c
                if w == cat:
                    dtotal_classified += 1
            for w in cats[c]["cwrong"]:
                if w == "NO CATEGORY":
                    print "IMPROPERLY CLASSIFIED CAVNAR: " + c
                if w == cat:
                    ctotal_classified += 1
        # indicate that all the texts incorrectly categorized
        # to be in this category are in this category
        # print "TOTAL CLASSIFIED: " + cat + ", " + str(ctotal_classified)
        cats[cat]["ccat"] = ctotal_classified
        cats[cat]["dcat"] = dtotal_classified
        if dtotal_classified == 0:
            cats[cat]["dprecision"] = 0
        else:
            cats[cat]["dprecision"] = cats[cat]["dright"]/dtotal_classified
        if ctotal_classified == 0:
            cats[cat]["cprecision"] = 0
        else:
            cats[cat]["cprecision"] = cats[cat]["cright"]/ctotal_classified

    #compute fmeasure
    for cat in cats:
        dprec = cats[cat]["dprecision"]
        drec = cats[cat]["drecall"]
        cprec = cats[cat]["cprecision"]
        crec = cats[cat]["crecall"]
        if dprec + drec == 0:
            cats[cat]["dfmeasure"] = 0
        else:
            cats[cat]["dfmeasure"] = 2 * (dprec * drec) / (dprec + drec)
        if cprec + crec == 0:
            cats[cat]["cfmeasure"] = 0
        else:
            cats[cat]["cfmeasure"] = 2 * (cprec * crec) / (cprec + crec)
    print "DAMASHEK HOBBIES: "
    #print cats["hobbies"]["dwrong"]
    log = open("results.csv", "w")
    print >>log, "category, Method, Accuracy, Precision, Recall, Fmeasure, Categorized"
    for cat in cats:
        print >>log, cat + ", Damashek, " + str(cats[cat]["daccuracy"]) + ", " + str(cats[cat]["dprecision"]) + ", " + str(cats[cat]["drecall"]) + ", " + str(cats[cat]["dfmeasure"]) + ", " + str(cats[cat]["dcat"])
        print >>log, ", Cavnar, " + str(cats[cat]["caccuracy"]) + ", " +  str(cats[cat]["cprecision"]) + ", " + str(cats[cat]["crecall"]) + ", " + str(cats[cat]["cfmeasure"]) + ", " + str(cats[cat]["ccat"])
    log.close()
    print "Damashek Accuracy: " + str(cos_correct/total)
    print "Damashec Wrong: " + str(total - cos_correct)
    print "Cavnar Accuracy: " + str(oop_correct/total)
    print "Cavnar Wrong: " + str(total - oop_correct)
    print "Total Classified: " + str(total)

def categorize_cat(category):
    print "categorizing: " + category
    total = 0
    count = 0
    cos_correct = 0
    oop_correct = 0
    shared = 0
    cos_false_negative = []
    oop_false_negative = []
    for text in brown.fileids(category)[10:]:
        # only compare the first five files after the training set
        if count >= COMPARISON_SET:
            count = 0
            break
        
        # indicate that we have compared one more file
        count += 1
        total += 1

        #categorize this text
        cos, oop = categorize(text)
        # check to see if the cos distance or out of place distance categorized
        # it correctly
        if cos == category:
            # cos distance correctly classified
            cos_correct += 1
        else:
            # cos distance incorrectly classified
            cos_false_negative.append(cos)
        if oop == category:
            # out of place distance correctly classified
            oop_correct += 1
            if cos == category:
                # both classified correctly
                shared += 1
        else:
            # out of place distance incorrectly classified
            oop_false_negative.append(oop)
    return (category, cos_correct, oop_correct, total, cos_false_negative, oop_false_negative)

 
def categorize(text):
    dcounts = {}
    dcounts = generate_text(text, dcounts, "damashek")
    ccounts = {}
    ccounts = generate_text(text, ccounts, "cavnar")
    cosine_estimate = "NO CATEGORY"
    cosine_min = sys.maxint
    out_of_place_estimate = "NO CATEGORY"
    out_of_place_min = sys.maxint
    #print "Beginning Catigorizations For " + text
    for category in CATAGORIES:
        # only use ngrams of length 5 for cosine measure
        # cos_counts = { i: j for i, j in ccounts.iteritems() if len(i) == 5 }
        tcos = cosine_measure(DDICT[category], dcounts)
        if tcos < cosine_min:
            cosine_estimate = category
            cosine_min = tcos
        tout = out_of_place_measure(CDICT[category], ccounts)
        if tout < out_of_place_min:
            out_of_place_estimate = category
            out_of_place_min = tout
    return (cosine_estimate, out_of_place_estimate)

def cosine_measure(template, sample):
    """ cosine_measure: compares the difference between template
        and sample dictionaries by comparing the cosine difference """
    # make template and sample have the same keys
    # Then sort in the same order
    for key in template:
        if key not in sample:
            sample[key] = 0

    for key in sample:
        if key not in template:
            template[key] = 0

    tvector = sorted(template.iteritems(), key=itemgetter(0), reverse=True)
    svector = sorted(sample.iteritems(), key=itemgetter(0), reverse=True)
    # turn relative frequencies into frequencies
    # make unit vectors
    ttotal = sum(tup[1] for tup in tvector)
    stotal = sum(tup[1] for tup in svector)
    tfreqs = [tup[1]/ttotal for tup in tvector]
    sfreqs = [tup[1]/stotal for tup in svector]
    return cosine_dist(tfreqs, sfreqs)

def cosine_dist(vect1, vect2):
    mu1 = 1/len(vect1)# * sum(vect1)
    mu2 = 1/len(vect2)# * sum(vect2)
    numerator = 0 #sum(x * x - mu1 for x in vect1)
    sumsqr1 = 0 #sum(x * x - mu2 for x in vect2)
    sumsqr2 = 0 #sum((x - mu1) * (y - mu2) for x in vect1 for y in vect2)
    for i in xrange(0, len(vect1)):
        x = vect1[i]; y = vect2[i]
        numerator += (x) * (y) 
        sumsqr1 += x*x
        sumsqr2 += y*y
    dist =  math.acos(numerator/math.sqrt(sumsqr1*sumsqr2))
    return dist 

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
        if svector[i][0] in keyindex:
            index = keyindex[svector[i][0]]
            dist += abs(index - i)
        else:
            # TODO: What if ngram is not present in svector
            dist += abs(len(tvector))
    return dist

def opendict():
    global DDICT, CDICT
    DDICT = pickle.load(open('ddict.pk1', 'rb'))
    CDICT = pickle.load(open('cdict.pk1', 'rb'))
def generate():
    global CDICT, DDICT
    for category in CATAGORIES:
        # skip the two categories with too little information
        CDICT[category] = generate_corpus(brown.fileids(categories=category), "cavnar")
        DDICT[category] = generate_corpus(brown.fileids(categories=category), "damashek")
    dict_file = open('cdict.pk1', 'wb')
    pickle.dump(CDICT, dict_file)
    dict_file.close()
    dict_file = open('ddict.pk1', 'wb')
    pickle.dump(DDICT, dict_file)
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
def generate_corpus(corpus, method):
    counts = {}
    max_training_items = 0
    for text in corpus:
        if max_training_items >= TRAINING_SET:
            break
        max_training_items += 1
        counts = generate_text(text, counts, method)
    return counts

def generate_text(text, counts, method):
    for word in brown.words(text):
        if re.match(PUNCT, word):
            continue
        counts = generate_ngrams(word.lower(), counts, method)
    return counts


def generate_ngrams(string, counts, method):
    global SMALLEST_NGRAM, LARGEST_NGRAM
    lower = SMALLEST_NGRAM
    upper = LARGEST_NGRAM
    if method == "damashek":
        lower = LARGEST_NGRAM - 2
    
    # pad string with upper-1 spaces on either side
    if method == "cavnar":
        string = " "*(upper-1) + string + " "*(upper-1)
    if method == "damashek":
        string = string
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
            elif gram in STOPWORDS:
                continue
            elif gram in counts:
                counts[gram] += 1
            else:
                counts[gram] = 1
    return counts


if __name__ == "__main__":
    generate()
    #opendict()
    categorize_all()
