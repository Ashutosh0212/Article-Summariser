import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def read_art(file_name):
    file = open(file_name, "r",errors="ignore")
    sentences = []
    while(True):
        filedata = file.readline()
        if len(filedata)==0:
            break
        # print(filedata)
        article = filedata.split(".")
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    # print(sentences)
    sentences.pop()
    return sentences


def sentence_similarity(sent1, sent2, stop_words):
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    if stop_words is None:
        stopwords = []
    else:
        for w in sent1:
            if w in stop_words:
                continue
            vector1[all_words.index(w)] += 1

        for w in sent2:
            if w in stop_words:
                continue
            vector2[all_words.index(w)] += 1

    # print(1 - cosine_distance(vector1, vector2))

    return 1-cosine_distance(vector1, vector2)


def gen_similqarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            else:
                x=sentence_similarity(sentences[idx1], sentences[idx2],stop_words)
                # print(x)
                similarity_matrix[idx1][idx2] = x
    return similarity_matrix


def generate_summary(file_name):
    stop_words = stopwords.words("english")
    # print(stop_words)
    summarize_text = []
    sentences = read_art(file_name)
    # print(sentences)
    sentence_similarity_matrix = gen_similqarity_matrix(sentences, stop_words)
    sentences_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentences_similarity_graph)
    # print(scores)
    ranked_sentence = sorted(([scores[i], s] for i, s in enumerate(sentences)), reverse=True)
    # print(ranked_sentence)
    # print("HEY")
    # for i in range(len(ranked_sentence)):
    for i in range(3):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        print ("--> ",ranked_sentence[i][0]," ".join(ranked_sentence[i][1]))
    # print(*summarize_text)


generate_summary("in.txt")
