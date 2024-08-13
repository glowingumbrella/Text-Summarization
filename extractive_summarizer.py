# TF-IDF vectorization with cosine similarity method adapted from https://medium.com/@krause60/using-cosine-similarity-to-build-a-python-text-summarization-tool-d3c8228549bf

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation

def summarize(review, n = 1):
    #tokenize, remove all stop words, and put it into a tf-idf vectorizer
    stops = list(set(stopwords.words("english"))) + list(punctuation)
    sent_list = sent_tokenize(review)
    vectorizer = TfidfVectorizer(stop_words = stops)
    trsfm = vectorizer.fit_transform(sent_list)

    #get the similarity of each pair of sentences
    similarities = cosine_similarity(trsfm,trsfm)
    
    #take average of each of the cosine similarities for each sentence
    avgs = [i.mean() for i in similarities]

    #take n highest average cosine similarity sentence and return
    sorted_sim = sorted(list(enumerate(avgs)),key = lambda x: x[1], reverse = True)
    summary = ""
    for i in range(n):
        summary += sent_list[sorted_sim[i][0]].replace("\\n"," ") + " "
    return summary

