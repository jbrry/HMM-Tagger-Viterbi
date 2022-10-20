"""
Implementation of the Viterbi Algorithm (recursive definition) from Chapter 2
of Michael Collins' NLP Lecture notes: http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf
"""

import sys
import util

from typing import Dict


class ViterbiTagger:
    def __init__(
        self,
        count_file: str,
        dev_file: str,
        states: Dict[str, int] = None,
        word_states: Dict[str, int] = None,
        bigram_count: Dict[str, int] = None,
        trigram_count: Dict[str, int] = None,
    ):

        self.count_file = count_file
        self.dev_file = dev_file

        self.states, self.word_states = util.build_emission_counts(self.count_file)
        self.bigram_counts, self.trigram_counts = util.build_transition_counts(
            self.count_file
        )

    def collate_sentences(self):
        sentences = []
        current_sentence = []
        dev_file = open(self.dev_file, "r")
        for line in dev_file:
            # new sentence
            if line.isspace():
                sentences.append(current_sentence)
                current_sentence = []
            else:
                current_sentence.append(line.strip().split("\t")[0])
        return sentences

    def get_tags(self, index):
        """
        Returns the permissiable tags for a given index in the sentence.
        If the index is -1 or 0, the only available tag is the start one '*'.
        Otherwise, we return a list of the tags in the states 'S'.
        """
        # before the beginning of the sentence.
        if index <= 0:
            return ["*"]
        # within the range of the sentence.
        elif index >= 1:
            return list(self.states.keys())

    def get_transition_prob(self, trigram, bigram):
        """Calculate transition probability."""
        try:
            transition_prob = self.trigram_counts[trigram] / self.bigram_counts[bigram]
        except KeyError:
            transition_prob = 0
        return transition_prob

    def get_emission_prob(self, word_state, state):
        """Calculate emission probability."""
        try:
            emission_prob = self.word_states[word_state] / self.states[state]
        # the word does not appear with that tag.
        except KeyError:
            emission_prob = 0
        return emission_prob

    def run(self, sentences, normalized_sentences):
        for i, sentence in enumerate(normalized_sentences):
            sentence_length = len(sentence)
            # + 2 to account for start states: (*, *), and we want to run until y_n+1.
            p = [{} for k in range(sentence_length + 2)]
            b = [{} for k in range(sentence_length + 1)]  # Store backpointers

            # set first elements
            p[0]["* *"] = 1
            b[0]["* *"] = "*"
            last_best_score = 0

            # For any position in the sentence k,
            # we consider all possible tag pairs (u, v)
            # which are possible at positions k-1 and k, respectively.
            # We then compute p[k][u v] using the max w
            # which occured at position k-2.
            for k in range(
                1, len(sentence) + 2
            ):  # +2 because we start at position 1 and also want to calculate the score of transitioning to STOP.
                for u in self.get_tags(k - 1):
                    for v in self.get_tags(k):
                        # set initial value for table
                        p[k][f"{u} {v}"] = 0
                        # This covers y1 ... yn, i.e. the tokens of the sentence
                        if k <= len(sentence):
                            word = sentence[k - 1]  # -1 because k starts from 1
                            word_state = " ".join([word, v])
                            emission_prob = self.get_emission_prob(word_state, v)

                            # Search over all possible values for w (at position-2), and return the maximum.
                            best_w_score = 0
                            for w in self.get_tags(k - 2):
                                trigram = " ".join([w, u, v])
                                bigram = " ".join([w, u])
                                transition_prob = self.get_transition_prob(
                                    trigram, bigram
                                )

                                # The recursive definition of Viterbi uses the score of the previous timestep.
                                prev = p[k - 1][bigram]
                                score = prev * transition_prob * emission_prob
                                if score >= best_w_score:  # = to allow 0 prob words
                                    best_w_score = score
                                    best_w = w
                            # Only record the best after trying all values of w.
                            p[k][f"{u} {v}"] = best_w_score
                            b[k][f"{u} {v}"] = best_w

                        # Calculate probability of transitioning to the STOP symbol for position yn+1.
                        else:
                            # Edge case: we won't have n*n tag keys to look up, so we will have to fall back to *.
                            if len(sentence) == 1:
                                u = "*"
                            trigram = " ".join([u, v, "STOP"])
                            bigram = " ".join([u, v])
                            transition_prob = self.get_transition_prob(trigram, bigram)

                            # k-1 is equivalent to the last token
                            score = p[k - 1][bigram] * transition_prob
                            p[k][f"{u} {v}"] = score

            # Sort the last index (of u v pairs) by highest probability.
            sorted_last = sorted(list(p[-1].items()), key=lambda x: x[1], reverse=True)
            # Take the first element in the sorted list, and the first element of the (bigram, prob) tuple.
            backtrack_bigram = sorted_last[0][0]

            decoded_tags = []
            last_tag = backtrack_bigram.split(" ")[1]
            second_last_tag = backtrack_bigram.split(" ")[0]
            # Edge case: if the sentence is length 1, we won't have all tags as keys, just the * one.
            if second_last_tag == "*":
                decoded_tags.append(last_tag)
                return decoded_tags

            decoded_tags.insert(0, last_tag)
            decoded_tags.insert(0, second_last_tag)

            # Backtrack over the sentence to recover the arg-max tags based on the
            # bigram value of (y+1 y+2)
            for k in range((len(b) - 1) - 2, 0, -1):
                best = b[k + 2][backtrack_bigram]
                decoded_tags.insert(0, best)
                backtrack_bigram = " ".join([best, backtrack_bigram.split()[0]])

            # Write output
            for word, tag in zip(sentences[i], decoded_tags):
                print(f"{word}\t{tag}")
            print()

        return decoded_tags


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python viterbi.py <count_file> <dev_file>")
        sys.exit(2)
    count_file = sys.argv[1]
    dev_file = sys.argv[2]

    tagger = ViterbiTagger(count_file, dev_file)
    sentences = tagger.collate_sentences()
    normalized_sentences = util.normalize_sentences(sentences)
    tagger.run(sentences, normalized_sentences)
