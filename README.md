### Trigram Hidden Markov Model with Viterbi Decoding
This repository contains an implementation of the Viterbi Algorithm (recursive definition) from Chapter 2 of Michael Collins' NLP Lecture notes: http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf.

#### Download data
```bash
./scripts/download_ud_data.sh UD_English-LinES en_lines
```

#### Obtain count frequencies for the HMM model
```bash
python count_freqs.py data/UD_English-LinES/en_lines-ud-train.pos output/vocab.txt > output/en_lines-ud-train.counts
```


#### Run the Viterbi Algorithm on a dev/test file
```bash
python viterby.py output/en_lines-ud-train.counts data/UD_English-LinES/en_lines-ud-dev.pos > output/en_lines-ud-dev.pred
```

#### Evaluate against a gold-standard file
```bash
python eval_tagger.py data/UD_English-LinES/en_lines-ud-dev.pos output/en_lines-ud-dev.pred
```
This should achieve a model accuracy of `90.20`.
