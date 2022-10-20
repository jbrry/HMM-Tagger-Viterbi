__author__ = "Daniel Bauer <bauer@cs.columbia.edu>"
__date__ = "$Sep 29, 2011"

# modified by James Barry

import sys


def corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    tagfield = 1

    try:
        while l:
            line = l.strip()
            if line:  # Nonempty line
                # Extract information from line.
                # Each line has the format
                # word tag
                fields = line.split("\t")
                ne_tag = fields[tagfield]
                word = " ".join(fields[:tagfield])
                yield word, ne_tag
            else:  # Empty line
                yield (None, None)
            l = corpus_file.readline()
    except IndexError:
        sys.stderr.write("Could not read line: \n")
        sys.stderr.write(f"\n{line}")
        sys.exit(1)


class Evaluator(object):
    """
    Stores correct/incorrect counts. 
    """

    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.total = 0

    def compare(self, gold_standard, prediction):
        """Compare the prediction against a gold standard."""
        for gold_input, pred_input in zip(gold_standard, prediction):
            gs_word, gs_tag = gold_input[0], gold_input[1]
            pred_word, pred_tag = pred_input[0], pred_input[1]

            # Make sure words in both files match up
            if gs_word != pred_word:
                sys.stderr.write(f"Could not align gold standard and predictions.\n")
                sys.stderr.write(
                    f"Gold standard: {gs_word}  Prediction file: {pred_word}\n"
                )
                sys.exit(1)

            if gs_word is not None:
                if gs_tag == pred_tag:
                    self.correct += 1
                else:
                    self.incorrect += 1
                self.total += 1

    def print_scores(self):
        """
        Prints model accuracy.
        """
        print(f"Model accuracy: {self.correct/self.total * 100:.2f}")


def usage():
    sys.stderr.write(
        """
    Usage: python eval_tagger.py <gold_file> <prediction_file>
        Evaluate the tagger output in <prediction_file> against
        the gold standard in <gold_file>. Outputs accuracy.\n"""
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    gold = open(sys.argv[1], "r")
    pred = open(sys.argv[2], "r")
    gs_iterator = corpus_iterator(gold)
    pred_iterator = corpus_iterator(pred)
    evaluator = Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()
