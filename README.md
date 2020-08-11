## Phonetic spellchecker

This program takes an English-y word and returns the dictionary word(s) with most similar pronunciation. It catches a subset of misspellings invisible to traditional spellcheckers — namely those that “look different, but sound the same”.

### Usage

`python spellcheck.py`

### Example run

	Enter ‘q’ to quit.

	Enter a word: akshulie
	Did you mean: actually

	Enter a word: kuzz-zyn
	Did you mean: cousin

	Enter a word: zookeeny
	Did you mean: zucchini

	Enter a word: q


### How it works

The idea is simple: map lexical sequences (words) to phonemic sequences (pronunciations), then look up the nearest phonemic sequence in the dictionary.

We map words to pronunciations with a sequence-to-sequence transformer trained on the CMU Pronouncing Dictionary. Sequences are compared with Levenshtein distance.

### Dependencies

The following packages need to be installed:

1. `nltk` — for the `words` corpus.
2. `editdistance` — fast C implementation of Levenshtein distance.
3. `torch` — neural network library. Version 1.6.0 or higher.
4. `torchtext` — used to process data. Note: the `Example` and `Field` classes will soon be deprecated.

### Optimizations

There is lots of room for experimentation and improvement. 

* Sequence translation is greedy — could be improved with beam search. 
* Edit distance search is exhaustive and likely too slow for some applications (see <https://norvig.com/spell-correct.html> for alternatives).
* Almost no hyperparameter search conducted while training the model. Doubling dimensionality (from 128 to 256) yields about 2% accuracy improvement but at the cost of quadrupling size. The bundled `seq2seq.pt` file is ~5 MB and >20 MB seemed excessive.
* Vanilla transformer model — no fancy modifications.
* Syllabic information stripped away for simplicity — could be incorporated.
* Label smoothing may also improve accuracy.

### For your own use

To incorporate this work into a larger system, simply instantiate a `Recommender` and call `recommend()`, which takes a word and returns a list of matching words. 

**Disclaimer:** Few checks on malformed input. Also, `.py` files not organized as importable modules.

### Acknowledgements

This small program would not be possible without the decades-long work that has gone into the CMU Pronouncing Dictionary.

I would also like to thank [Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) and [Aladdin Persson](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py), whose transformer-from-scratch code I copied extensively.