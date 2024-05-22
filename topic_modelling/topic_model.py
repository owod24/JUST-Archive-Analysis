from gensim import corpora, models


def create_dictionary_and_corpus(processed_texts):
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    return dictionary, corpus


def apply_lda_model(corpus, dictionary, num_topics=10, passes=15):
    lda_model = models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=passes
    )
    return lda_model


def print_topics(lda_model, num_words=10):
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)
