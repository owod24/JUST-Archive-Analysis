from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd


def analyze_keywords(processed_texts, keywords):
    all_words = [word for text in processed_texts for word in text]
    word_count = Counter(all_words)
    keyword_frequencies = {keyword: word_count[keyword] for keyword in keywords}
    return keyword_frequencies


def plot_keyword_frequencies(keyword_frequencies, title="Keyword Frequencies"):
    plt.bar(keyword_frequencies.keys(), keyword_frequencies.values(), color="skyblue")
    plt.xlabel("Keywords")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def extract_keyword_contexts(processed_texts, keywords):
    sentences = nltk.sent_tokenize(
        " ".join([" ".join(text) for text in processed_texts])
    )
    context_dict = {keyword: [] for keyword in keywords}
    for sentence in sentences:
        for keyword in keywords:
            if keyword in nltk.word_tokenize(sentence):
                context_dict[keyword].append(sentence)
    return context_dict


def save_contexts_to_csv(context_dict, filename="keyword_contexts.csv"):
    context_df = pd.DataFrame(
        [
            (keyword, context)
            for keyword in context_dict
            for context in context_dict[keyword]
        ],
        columns=["Keyword", "Context"],
    )
    context_df.to_csv(filename, index=False)
    return context_df
