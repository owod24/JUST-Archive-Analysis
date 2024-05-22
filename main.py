import os

from keyword_analysis.keyword_context import (
    analyze_keywords,
    extract_keyword_contexts,
    plot_keyword_frequencies,
    save_contexts_to_csv,
)
from preprocessing.preprocess import preprocess
from preprocessing.read_pdf import read_pdf
from topic_modelling.topic_model import (
    apply_lda_model,
    create_dictionary_and_corpus,
    print_topics,
)

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


keywords = [
    "marginalized",
    "indigenous",
    "gender",
    "black",
    "equality",
    "inclusion",
    "oppression",
    "colonization",
    "culture",
    "sexuality",
    "race",
    "ethnicity",
]


# keyword_frequencies = analyze_keywords(processed_texts, keywords)
# plot_keyword_frequencies(keyword_frequencies, title="Word Frequency - Vol 2(2009)")


# TODO: Csv file is incoherent and not readable. What are some possible fixes?
# context_dict = extract_keyword_contexts(processed_texts, keywords)
# context_df = save_contexts_to_csv(context_dict)
#
# Topic Modeling
dictionary, corpus = create_dictionary_and_corpus(processed_texts)
lda_model = apply_lda_model(corpus, dictionary, num_topics=10, passes=15)
print_topics(lda_model)
