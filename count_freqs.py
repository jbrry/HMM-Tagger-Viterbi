__author__ = "Daniel Bauer <bauer@cs.columbia.edu>"
__date__ = "$Sep 12, 2011"

# modified by James Barry

import sys
from collections import defaultdict
import math


def create_vocab(train_path, vocab_path):
    """
    Creates and writes words in a vocabulary to an output path, provided
    the word has been seen at least min-count number of times.
    """
    token_count = 0
    min_count = 2
    vocab_counter = defaultdict(int)
    vocab = []

    # create vocabulary from training file
    with open(train_path, "r") as fi:
        # print("Generating vocabulary\n")
        sys.stderr.write("Generating vocabulary\n")
        for line in fi:
            line = line.strip()
            if line:
                word_tag = line.split("\t")
                word = word_tag[0]
                tag = word_tag[1]
                count = vocab_counter[word]
                vocab_counter[word] = count + 1
                token_count += 1
            else:  # Empty line
                continue
    fi.close()

    # write out vocab
    with open(vocab_path, "w") as fo:
        sys.stderr.write("Writing vocabulary\n")
        for word, count in vocab_counter.items():
            if count >= min_count:
                fo.write(word + "\n")
                vocab.append(word)
    fo.close()

    before = len(vocab_counter.keys())
    after = len(vocab)
    removed = before - after
    sys.stderr.write(
        f"Out of {before} tokens, {removed} were removed for having a count less than {min_count}\n"
    )

    return vocab


# Count n-gram frequencies in a CoNLL data file and write counts to
def simple_conll_corpus_iterator(corpus_file, vocab):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """

    with open(corpus_file, "r") as fi:
        for line in fi:
            line = line.strip()
            if line:
                word_tag = line.split("\t")
                word = word_tag[0]
                if word not in vocab:
                    word = "<unk>"
                tag = word_tag[1]
                yield word, tag
            else:  # Empty line
                yield (None, None)
    fi.close()


def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, tag) tuples.
    """
    current_sentence = []
    for l in corpus_iterator:
        if l == (None, None):
            if current_sentence:
                yield current_sentence
                current_sentence = []
            else:
                sys.stderr.write("WARNING: Got empty input file/stream.\n")
                raise StopIteration
        else:
            current_sentence.append(l)

    if current_sentence:  # If the last line was blank, we're done
        yield current_sentence  # Otherwise when there is no more token
        # in the stream return the last sentence.


def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    sent_iterator is a generator object whose elements are lists
    of tokens.
    Outputs ngrams like: (('enjoyed', 'VERB'), ('muttering', 'VERB'), ('nonsense', 'NOUN'))
    """
    for sent in sent_iterator:
        # Add boundary symbols to the sentence
        # n-1 to account for 2 start symbols needed for a trigram model
        w_boundary = (n - 1) * [(None, "*")]  # No word and start tag
        w_boundary.extend(sent)
        w_boundary.append((None, "STOP"))  # No word and stop tag

        # Then extract n-grams
        ngrams = (tuple(w_boundary[i : i + n]) for i in range(len(w_boundary) - n + 1))
        for n_gram in ngrams:  # Return one n-gram at a time
            yield n_gram


class HMM(object):
    """
    Stores counts for n-grams and emissions.
    """

    def __init__(self, n=3):
        assert n >= 2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

    def train(self, corpus_file, vocab):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = get_ngrams(
            sentence_iterator(simple_conll_corpus_iterator(corpus_file, vocab)), self.n
        )
        for ngram in ngram_iterator:
            # Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert (
                len(ngram) == self.n
            ), f"ngram in stream is {len(ngram)}, expected {self.n}"

            tagsonly = tuple(
                [tag for word, tag in ngram]
            )  # retrieve only the tags (second element)
            for i in range(2, self.n + 1):  # Count tag 2-grams..n-grams
                self.ngram_counts[i - 1][tagsonly[-i:]] += 1

            if ngram[-1][0] is not None:  # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1  # count 1-gram
                self.emission_counts[ngram[-1]] += 1  # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None:  # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1, 2, 3]):
        """
        Writes counts to the output file object.
        Format:
        """
        # First write counts for emissions
        for word, tag in self.emission_counts:
            output.write(f"{self.emission_counts[(word, tag)]} WORDTAG {tag} {word}\n")
        # Then write counts for all ngrams
        for n in printngrams:
            for ngram in self.ngram_counts[n - 1]:
                ngramstr = " ".join(ngram)
                output.write(f"{self.ngram_counts[n-1][ngram]} {n}-GRAM {ngramstr}\n")


if __name__ == "__main__":
    if (
        len(sys.argv) != 3
    ):  # Expect exactly 2 arguments: the training data file and output vocab file.
        print("python count_freqs.py [input_file] [vocab_file] > [output_file]")
        sys.exit(2)
    try:
        train_file = sys.argv[1]
        vocab_file = sys.argv[2]
    except IOError:
        sys.stderr.write(f"ERROR: Cannot read inputfile {arg}.\n")
        sys.exit(1)

    # Create vocab
    vocab = create_vocab(train_file, vocab_file)
    # Initialize a trigram counter
    counter = HMM(3)
    # Collect counts
    counter.train(train_file, vocab)
    # Write the counts
    counter.write_counts(sys.stdout)