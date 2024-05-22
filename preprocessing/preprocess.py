import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
