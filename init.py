import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader

# TODO: Understand the utility of each of the above packages


# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


domain_stopwords = set(
    [
        "marginalized",
        "equity",
        "inclusive",
        "gender",
        "sexuality",
        "race",
        "ethnicity",
        "disability",
        "accessibiity",
        "inclusive",
        "practise",
    ]
)


# Preprocessing function
# TODO: Understand the steps taken to preprocess the text
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union(domain_stopwords)
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return tokens


# Function to read PDF files
def read_pdf(file_path):
    reader = PdfReader(open(file_path, "rb"))
    text = ""
    for page_num in range(reader.get_num_pages()):
        text += reader.get_page(page_num).extract_text()
    return text


# Path to the folder containing documents
folder_path = os.path.expanduser("~/Downloads/JUST Papers/Vol 2 - 2009/")

# List to hold all the processed texts
processed_texts = []
document_names = []

# Iterate over each document in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        # Read the content of the pdf file
        content = read_pdf(os.path.join(folder_path, filename))
        # Preprocess the content
        processed_text = preprocess(content)
        processed_texts.append(processed_text)
        document_names.append(filename)


# Flatten the list of processed texts
all_words = [word for text in processed_texts for word in text]

keywords = [
    "marginalized",
    "indigenous",
    "gender",
    "black",
    # "people of color",
    # "colored",
    "equality",
    "inclusion",
    "oppression",
    "colonization",
    "culture",
    "sexuality",
    "race",
    "ethnicity",
]

word_count = Counter(all_words)
keyword_frequencies = {keyword: word_count[keyword] for keyword in keywords}

plt.bar(keyword_frequencies.keys(), keyword_frequencies.values(), color="skyblue")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.title("Word Frequency - Vol 2(2009)")
plt.show()


# # Create a dictionary and corpus
# # TODO: Understand what these data structures are and their function
# dictionary = corpora.Dictionary(processed_texts)
# corpus = [dictionary.doc2bow(text) for text in processed_texts]
#
# # Apply LDA model
# num_topics = 10
# lda_model = gensim.models.LdaModel(
#     corpus, num_topics=num_topics, id2word=dictionary, passes=15
# )

# topics = lda_model.print_topics(num_words=10)
# topic_labels = {}
# for i, topic in topics:
#     print(f"Topic {i}: {topic}")
#     # Manually assign a label based on the terms in the topic
#     # Example labels - add your own labels
#     if i == 0:
#         topic_labels[i] = "Inuit Art and Culture"
#     elif i == 1:
#         topic_labels[i] = "Black Canadians"
#     elif i == 2:
#         topic_labels[i] = "LGBTQ+ Communities"
#     elif i == 3:
#         topic_labels[i] = "People with Disabilities"
#     elif i == 4:
#         topic_labels[i] = "Women and Girls"
#     elif i == 5:
#         topic_labels[i] = "Immigrants and Refugees"
#     elif i == 6:
#         topic_labels[i] = "Religious Minorities"
#     elif i == 7:
#         topic_labels[i] = "Low-Income Populations"
#     elif i == 8:
#         topic_labels[i] = "Youth and Elderly Populations"
#     elif i == 9:
#         topic_labels[i] = "Intersectional Groups"
#     else:
#         topic_labels[i] = f"Topic {i}"
#
# # Get the topic distribution for each document
# doc_topics = lda_model.get_document_topics(corpus, minimum_probability=0.0)
#
# # Threshold for determining significant topic presence in a document
# threshold = 0.5
#
# # Identify relevant documents
# relevant_docs = []
#
# for i, doc in enumerate(doc_topics):
#     for topic_num, proportion in doc:
#         if proportion > threshold and topic_num in topic_labels:
#             relevant_docs.append(
#                 (document_names[i], topic_labels[topic_num], proportion)
#             )
#
# # Print relevant documents
# print("\nRelevant Documents:")
# for doc_name, topic_label, proportion in relevant_docs:
#     print(f"Document: {doc_name}, Topic: {topic_label}, Proportion: {proportion:.2f}")

# Print the topics
# topics = lda_model.print_topics(num_words=10)
# for topic in topics:
#     print(topic)
