import math
import string

import nltk
import sys
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    nltk.download('stopwords')
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    directory_path = os.path.join(directory)
    filenames = os.listdir(directory_path)
    files = {}
    for file in filenames:
        with open(os.path.join(directory_path, file)) as text:
            files[file] = text.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    tokens = nltk.word_tokenize(document)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = []
    for token in tokens:
        if token not in string.punctuation and token not in stop_words:
            words.append(token)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    idfs = dict()
    for document in documents:
        words.update(documents[document])
    for word in words:
        frequency = sum(word in documents[document] for document in documents)
        idf = math.log(len(documents) / frequency)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = dict()
    for file in files:
        tf_idfs[file] = 0
        for word in query:
            term_frequency = 0
            for word_file in files[file]:
                if word == word_file:
                    term_frequency += 1
            tf_idfs[file] += term_frequency * idfs[word]

    l = list(tf_idfs.items())
    l.sort(key=lambda x: x[1], reverse=True)
    keys = [x[0] for x in l]
    return keys[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    tf_idfs = dict()
    for sentence in sentences:
        tf_idfs[sentence] = 0
        for word in query:
            if word in sentences[sentence]:
                tf_idfs[sentence] += idfs[word]
    tf_idfs_list = list(tf_idfs.items())
    tf_idfs_list.sort(key=lambda x: x[1], reverse=True)
    keys = [x[0] for x in tf_idfs_list]
    return keys[:n]


if __name__ == "__main__":
    main()
