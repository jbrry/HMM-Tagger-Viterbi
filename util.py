import sys
import re


# More strategies for handling unknown words can be added here
def normalize(word):
    word = "<unk>"
    return word

def normalize_sentences(sentences):
    """
    Checks to see if a word exists in the vocab file, and if not
    replaces the word with an <unk> form.
    """
    vocab = []
    with open("output/vocab.txt", "r") as vocab_file:
        for line in vocab_file:
            line = line.strip()
            vocab.append(line)

    normalized_sentences = []
    current_sentence = []
    for s in sentences:
        for word in s:
            if word not in vocab:
                word = normalize(word)
            current_sentence.append(word)
        normalized_sentences.append(current_sentence)
        current_sentence = []
    return normalized_sentences


def build_emission_counts(filename):
    counts = open(filename, "r")
    states = {}
    word_states = {}

    for line in counts:
        parts = line.strip().split(" ")
        identifier = parts[1]  # either <WORDTAG> or <N-GRAM>
        if "WORDTAG" in identifier:
            count = int(parts[0])
            tag = parts[2]
            word = parts[3]
            try:
                states[tag] = states[tag] + count
            except KeyError:
                states[tag] = count
            word_state = " ".join([word, tag])
            try:
                word_states[word_state] = word_states[word_state] + count
            except KeyError:
                word_states[word_state] = 0
                word_states[word_state] = word_states[word_state] + count

    counts.close()
    return states, word_states


def build_transition_counts(filename):
    counts = open(filename, "r")
    trigram_counts = {}
    bigram_counts = {}
    for line in counts:
        parts = line.strip().split(" ")
        count = int(parts[0])
        identifier = parts[1]
        if "2-" in identifier:
            prev2 = parts[2]
            prev1 = parts[3]
            bigram = " ".join([prev2, prev1])
            bigram_counts[bigram] = count
        elif "3-" in identifier:
            prev2 = parts[2]
            prev1 = parts[3]
            curr = parts[4]
            trigram = " ".join([prev2, prev1, curr])
            trigram_counts[trigram] = count

    counts.close()
    return bigram_counts, trigram_counts
