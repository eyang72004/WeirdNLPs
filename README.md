# WeirdNLPs 🧠📚 - A Very Janky and Weird Attempt at Building a Modern C++ NLP Library 

Hello, and Welcome to WeirdNLPs. This is my attempt at developing a full-stack natural language processing library in C++.
It also marks the first time I made an attempt to work with Python bindings via pybind11, as I had been inspired by modern NLP toolkits and tried to build it from scratch with jank, learning, and experimentation in mind.

This README I have attempted to create here is a basic guide to hopefully understanding and using the library across all 13 development phases, where I tried to delve into each and every one while trying to establish certain fundamental features and (more importantly) weirdness.
Here we will attempt to explore each function, file, design decision, and potential enhancements for hopeful future version to come.

## Table of Contents
1. Phase-by-Phase Details
    - Phase 1: Tokenization
    - Phase 2: Stemming
    - Phase 3: Lemmatization
    - Phase 4: Corpus
    - Phase 5: Vectorization
    - Phase 6: Embeddings
    - Phase 7: Syntax
    - Phase 8: Named Entity Recognition
    - Phase 9: Sentiment
    - Phase 10: Machine Learning Models
    - Phase 11: Deep Learning Models
    - Phase 12: Utils
    - Phase 13: Markov Chain

2. [Installation and Setup](#installation-and-setup)

3. [How to Use](#how-to-use)

4. [Development Goals](#development-goals)

5. [Known Quirks](#known-quirks)

6. [Future Work](#future-work)

---

## Phase-by-Phase Details

Each phase below corresponds to a certain module in the NLP pipeline.

### Phase 1: Tokenization

Implements whitespace-based and regex-based tokenizers for word-level processing.
The regex tokenizer captures alphanumeric word tokens, including simple contractions and hyphenated forms.
A sentence splitter is also included, using punctuation-based heuristics to extract complete sentences.
Hopefully these tokenizers output token vectors suitable for downstream analysis.

#### Potential Improvements??
- **Better Sentence Boundary Detection**: Current sentence splitting uses simple punctuation heuristics. It could fail for abbreviations (e.g., `Dr.`, `U.S.`, `e.g.`, etc.), and for sentences without proper punctuation (common in social media and/or informed text).
- **Unicode and Multilingual Support**: Current regex assumes English alphanumeric characters only. Support for Unicode letters, emojis, scripts in different languages, and right-to-left languages could be added.
- **Contractions and Special Symbols**: While basic contractions such as `didn't` are somewhat handled, more advanced normalization (e.g., `gonna`, `lemme`, etc.) are not completely handled. (I say "not completely handled" because I may have navigated my way somewhat in Phase 8: NER..)
- **Customizable Token Filters**: Allowing users to configure stopword removal, lowercasing, punctuation stripping, etc., could make tokenization more flexible and reusable across different domains.
- **Streaming Tokenization**: For memory efficiency, tokenization could be implemented as a generator or on-the-fly processor for long texts.

### Phase 2: Stemming

Attempted to implement a version of the Porter Stemming Algorithm from scratch, covering five steps (1a-1c, 2, 3, 4, 5).
The algorithm would seek to strip common English suffixes such as `-ing`, `-ed`, `-ness`, and `-zation`, reducing words to their crude stems in an effort to establish better generalization in NLP tasks.

The implementation tries to include these:
    - Plural reduction (e.g., `ponies -> poni`, `caresses -> caress`)
    - Past tense and gerund handling with cleanup logic (e.g., `running -> run`)
    - Double consonant reduction and simplification (e.g., `hopping -> hop`)
    - Final `-y` to `-i` conversion when preceded by a vowel (e.g., `happy -> happi`, `happily -> happi`)
    - Edge Case handling for stems such as `plast -> plaster` and `abl -> abil` via brute-force checks

In this version, I tried to avoid recursion in an effort to establish better clarity and performance, and designed it to be extendable and auditable; hopefully these can support language-specific suffix expansions and/or exception lists.
    - NOTE: Currently optimized for English only. May produce non-intuitive stems for proper nouns, acronyms, or compound terms.

#### Potential Improvements??

- Add multilingual suffix rules or pluggable rule sets for other languages
- Consider hybrid stemmer-lemmatizer model that uses rules when no lemma is found
- Integrate with vectorization in Phase 5 for stemming-aware tokenization pipelines

### Phase 3: Lemmatization

A basic dictionary-based lemmatizer that maps inflected words to their lemma forms using a flat `unordered_map<std::string, std::string>`.
Each entry in the lexicon corresponds to one (inflected -> base) mapping. 
If no lemma is found, then the original word is returned.
This was an attempt to make a somewhat simple, language-agnostic design -- ideal for extensibility but without morphological context (no POS disambiguation...hopefully later though).

#### Potential Improvements??

- **POS-Sensitive Lemmatization**: Currently, this design does not distinguish between verb and noun forms, I reckon (e.g., "saw" as a noun vs. past tense of "see").
- **Multiple Lemma Mappings**: Some inflected forms (e.g., `left`) can map to multiple lemmas depending on context (`leave` Vs. `left`).
- **Case Normalization**: Future versions may attempt to lowercase input tokens or use a case-insensitive map for better robustness.
- **Multilingual Support**: By swapping in a different lemma dictionary, I think that the design can theoretically support any language with morphological variation.
- **Hybrid Fallback**: If no lemma is found, then fall back to a rule-based or ML-based stemmer for unknown words.

### Phase 4: Corpus

Attempt to handle raw text ingestion and normalization.
The module would try to load plain-text files into memory, apply basic normalization (e.g., lowercasing, preserving inner hyphens and apostrophes) and expose utility functions to do these:
    - Retrieve the raw or normalized text
    - Split the raw text into sentences using the tokenization module
    - Tokenize the normalized text into words using regex

I tried to design it so that it simplifies preprocessing pipelines and serve as a backbone for later modules such as vectorization, sentiment, and syntax analysis.

#### Potential Improvements??

- **Custom Normalization Rules**: Allow injection of user-defined rules or regex transforms to better tailor preprocessing to specific domains (e.g., tweets, biomedical text, etc.).
- **Multi-Language Support**: Current normalization logic assumes English alphanumerics and ASCII-compatible output. Supporting non-English corpora may require Unicode-aware logic and language-specific rules.
- **Streaming/Token-By-Token File Reading**: Useful for handling very large files or real-time input streams.
- **Parallel Tokenization Support**: For long documents, allow thread-safe, chunk-based sentence splitting or word tokenization.

### Phase 5: Vectorization

Implements both Bag-of-Words (BoW) and TF-IDF vectorization, attempted from scratch.
'Vocabulary' class assigns a unique index to each word across the corpus using a hash map.
BoW vectorizer returns raw frequency counts for each term, while the TF-IDF vectorizer tries to calculate term weights using the smoothed formula of `log((1 + N) / (1 + df)) + 1`. 
Both vectorizers would return dense vectors where each dimension corresponds to a vocabulary term.
Prioritizes numerical correctness and simplicity, making it suitable for educational purposes and lightweight ML pipelines.

#### Potential Improvements??

- **Sparse Vector Support**: For larger corpora, storing vectors as sparse maps (e.g., `unordered_map<int, double>`) could potentially reduce memory overhead.
- **Normalization Option**: Add vector normalization (e.g., L2 Norm?) as an option for downstream models that assume unit-length inputs.
- **Inverse Vocabulary Access**: Maybe a reverse map like `index_to_word` could be helpful in decoding vector indices back into tokens.
- **n-gram Support**: Allow `Vocabulary::build()` to accept bi-grams and/or trigrams, increasing expressiveness at the cost of dimensionality.
- **TF Variant Toggle**: Allow binary (presence/absence) or sublinear scaling options for term frequency.

### Phase 6: Embeddings

This module attempts to support pre-trained static word embeddings (GloVe in this case) and provides these:
    - `contains(word)`: Attempts to check if a word has an embedding
    - `get_vector(word)`: Attempts to retrieve the embedding as a float vector
    - `cosine_similarity(w1, w2)`: Attempts to navigate through semantic similarity between words
    - `analogy(w1, w2, w3)`: Attempts to compute vector(w2) - vector(w1) + vector(w3)

I hope this is useful for measuring word similarity, semantic clustering, and/or analogy tasks in lightweight pipelines.

#### Potential Improvements??

- **Add-Word-to-Closest-Words Search**: Use cosine similarity to return top k most similar words to a target.
- **Support Multiple File Formats**: Allowing Word2Vec `.bin`, FastText `.vec`, etc.
- **Normalize Vectors On Load**: For faster cosine similarity computation.
- **Out-of-Vocab (OOV) Vector Handling**: Explore trainable or heuristic-based OOV embeddings.

### Phase 7: Syntax and POS Tagging

Attempts to implement a lexicon-based and rule-backed basic Part-of-Speech (POS) tagger.
- Loads word and POS-tag mappings from a lexicon file.
- If no tag is found, then attempt to apply simple regex-based heuristics:
    - `.*ing` → VBG (gerund)
    - `.*ed` → VBD (past tense)
    - `.*s` → NNS (plural noun)
    - `.*ly` → RB (adverb)
    - fallback → NN (noun)

#### Potential Improvements??

- **Lexicon Expansion**: Add more high-frequency words, function words (e.g., `and`, `or`, `but`, `if`), auxiliary/modal verbs, proper nouns, and punctuation to improve baseline accuracy.
- **Contraction Handling**: Currently, contractions such as `won't` and `didn't` may fall through the cracks. Preprocessing and/or explicit lexicon could fix this.
- **Context-Aware Tagging**: Introduce bigram or trigram models (e.g., HMMs or CRFs) to make tagging decisions based on surrounding words.
- **Multiple Tag Candidates**: Allow for probablistic tagging or top k tag outputs for ambiguous tokens (e.g., `mark` could be noun or verb).
- **Error Reporting or Unknown Logging**: Log unknown words or dfault-tagged words during inference for lexicon improvement or debugging.
- **Pattern Heuristic Refinement**: Improve heuristics for superlatives (`-est`), comparatives (`-er`), numerals, dates, and capitalized entities.
- **Language Adaptability**: Add support for multiple lexicons (presumably for other languages?) via optional locale argument in constructor.


### Phase 8: Named Entity Recognition (NER)

Attempts to implement a rule-based Named Entity Recognition module using a customizable lexicon file (`ner_lexicon.txt`).
The system identifies entity spans (e.g., `Barack Obama`) by performing up to trigram token matching, with optional support for multiword entries written using primarily underscores.

- Entity Matching:
    - Supports matching 1- to 4-token phrases
    - Input tokens are normalized (lowercased, filtered for punctuation) and, I would say, checked against a dictionary of known entities
    - When a match is found, the original case is restored for each token using the stored lexicon entry.

- Normalization Details:
    - Tokens and lexicon entries are normalized using these:
        - Lowercasing
        - Preserving alphanumeric characters, dashes (`-`), apostrophes (`'`), and presumably spaces
        - Filtering out all other punctuation
    - Lexicon matching is based on this normalized form, but output casing is preserved using a reverse-mapping to the lexicon's original entry

- This is one of those phases that I would say is rather underdeveloped to a great extent...
    - No contextual awareness -> tagging is based entirely on token strings and lexicon
    - Compound names (e.g., `Mark O'Connor`) must be entered as a single token or presumably some kind of known phrase
    - Incorrect token splits may result in missed matches
    - Middle-capital names (e.g., `Lebron` and `LeBron`) are for some reason not able to help themselves despite being typed verbatim in the lexicon

In principle, I attempted to make the module as interpretable and ideal to the best of my ability for deterministic tagging pipelines.
I did not create this with the intention of replacing NER models, but hopefully it would provide a useful foundation or fallback if you want more control over this...

#### Potential Improvements??

- **Entity Span Scoring And Disambiguation**: Currently, only the first matching span is tagged. Consider ranking candidate matches (e.g., prefer 3-grams over 2-grams only if both exist).
- **Case-Insenitive Matching with Smart Recovery**: Improve match logic for captialized variations such as `LeBron` vs. `Lebron` vs. `lebron`.
- **NER Label Granularity**: Introduce BILOU or BIO tagging schemes for compatibility with sequence labeling standards.
- **Heuristic Phrase Detection**: Add support for common title patterns such as `Dr.`, `President`, or maybe even `University of ____`.
- **Cross-Sentence Entity Recognition**: Add option to flag repeated names across multiple contexts (basic co-reference).
- **Regex-Based Triggers**: Enable inline rules such as (but are not limited to) these -> `\d{4}` → DATE, `Mr\. \w+` → PERSON, etc.
- **Confidence Estimation**: Include optional confidence flags (e.g., `high` if lexicon hit, `medium` if fallback, etc.).

### Phase 9: Sentiment

Attempts to implement a rule-based sentiment analysis system using a handcrafted lexicon.
Each entry in the lexicon maps a word to an integer sentiment score (e.g., `love 2`, `hate -2`).
The analyzer sums the score of individual tokens in a sentence and classifies the total score as these:
    - **Positive** (if score > 0)
    - **Negative** (if score < 0)
    - **Neutral** (if score = 0)

This approach requires no training or external dependencies.
I tried to make it provide a fast and interpretable baseline for evaluating sentiment polarity in short texts.

#### Potential Improvements??

- **Negation Handling**: Currently, the system does not consider words such as `not`, `never`, or `barely` as sentiment modifiers. This may skew results (e.g., `not great` may still count as positive).
- **Intensifier Modulation**: While words such as `very` and `extremely` are neutral, one could implement a sliding scale that amplifies adjacent word sentiment (e.g., `very happy` -> `+3`).
- **Sarcasm and Context**: Rule-based methods cannot detect sarcastic phrasing (`I LOVE waiting in line...`) or mixed polarity well.
- **Score Normalization**: Consider dividing the score by the number of sentiment-carrying words to make it length-independent.
- **Sentiment-Aware Tokenizer**: Some phrases such as `no good` or `badly done` span multiple words -- maybe could add phrase-level scoring or parse trees.
- **Cross-Lingual Lexicons**: Expand to multilingual support using locale-tagged lexicons.

### Phase 10: Classic Machine Learning Models

Attempted to introduce simple (and hopefully effective) classifiers for text classification tasks, operating on vectorized data (e.g., from BoW or TF-IDF).

#### Naive Bayes Classifier

Attempts to implement a multinomial Naive Bayes model with Laplace smoothing:
    - The `fit` attempts to take a list of BoW vectors and their associated class labels. Builds per-class word counts and computes log-probabilities for each word in each class.
    - The `predict` attempts to compute the posterior probability for each class and returns the one with the highest score.

#### Logistic Regression Classifier

Attempts to implement a binary logistic regression using stochastic gradient descent:
    - The `fit` attempts to iteratively optimize weight vectors using a standard cross-entropy over each document.
    - The `predict` attempts to apply the learned weights to a new vector and outputs 0 or 1 using sigmoid decision rule.
    - I tried to make it so that it would be somewhat compatible with BoW and TF-IDF from Phase 5.
    - I am not too sure, but convergence may need hyperparameter tuning (e.g., `lr`, `epochs`), especially for large vocabularies or unnormalized TF values.

I tried to bridge NLP with ML and serve it as a foundation for Phase 11 regarding Deep Learning Models...

#### Potential Improvements??

- **Multiclass Logistic Regression**: Current implementation is limited to binary classification. Maybe softmax regression (a generalization) could help with enabling multi-label outputs.
- **Regularization**: Weight decay (L2 penalty) maybe could help generalize better when vocab size is large. Could maybe be optionally passed into `fit(...)`.
- **Convergence Criteria**: Instead of fixed epochs, use dynamic stopping (e.g., loss difference threshold).
- **Weight Inspection / Model Dumping**: Methods such as `get_weights()` and `debug()` could expose internal model state for analysis or report.
- **Sparse Optimization**: Both classifiers assume dense BoW inputs (addressed primarily in the `test_ml_models.cpp` file). Sparse matrix support could maybe improve performance on large corpora.
- **Batching or Mini-Batch SGD**: Logistic Regression could maybe benefit from mini-batch updates for efficiency and stability.



### Phase 11: Deep Learning Models

Marks the first time I used anything pertaining to PyTorch (LibTorch here in this case).
Attempts to wrap a serialized `.pt` TorchScript model (compiled from Torch) for inference within C++. 

The `TorchScriptModel` class attempts to perform forward passes on preprocessed input IDs and returns raw logits as `std::vector<float>`. 
It also attempts to support class prediction by returning the highest logit.


#### Potential Improvements??

- **Input Encoding**: Current usage assumes that token IDs are already computed externally (e.g., BERT-style IDs). Maybe one could be able to integrate a tokenizer (e.g., WordPiece or SentencePiece) and vocabulary mapping for end-to-end interface.
- **Embedding Awareness**: While this phase wraps a classification head, the underlying model does not yet support embeddings or sequence processing (e.g., LSTM, Transformer). Future improvements could maybe load models with internal token embedding and attention modules.
- **Output Interpretation**: The logits are currently raw. Softmax postprocessing or thresholded classification for multi-label tasks could maybe improve versatility.
- **Model Introspection**: Logging layer names, weights, or architecture info (I suppose something along the lines of `model.dump_to_str()` maybe if exposed in TorchScript??) could potentially help with debugging or model selection.
- **Batch Inference Support**: The wrapper currently accepts a single vector. Supporting multiple samples (NxM input) could maybe improve inference throughput.



### Phase 12: Utilities

Attempts to include general-purpose helper functions that are used across the NLP pipeline.
The utilities would try to support text preprocessing and vector operations:
    - `to_lower()`: Attempts to convert text into lowercase
    - `remove_punctuation()`: Attempts to strip non-alphanumeric characters (except spaces hopefully)
    - `split_by_space()`: Attempts to tokenize a string by spaces
    - `cosine_similarity()`: Attempts to compute cosine similarity between two float vectors

I tried to make this module somewhat foundational for normalization, token filtering, and basic numerical comparisons.

#### Potential Improvements??

- **Unicode-Aware Normalization**: `to_lower()` and `remove_punctuation()` currently assume ASCII. In production NLP settings, maybe one could consider handling accented characters and multilingual input using libraries such as ICU or Boost.Locale.
- **Advanced Token Splitting**: `split_by_space()` could maybe be extended to split on Unicode whitespace or consider punctuation boundaries.
- **Angle Normalization for Cosine**: In rare high-dimensional use cases, returning `std::acos(similarity)`could maybe allow angular analysis between vectors.
- **Unit Tests**: Maybe add `catch2` and/or doctest-based unit tests to ensure long-term stability.
- **Whitespace Cleanup**: Maybe add a utility such as `strip()` and/or `squeeze_spaces()` to eliminate repeated spaces or leading/trailing whitespace.



### Phase 13: Markov Chains

Attempts to implement an nth-order Markov model for stochastic text generation.
Given a sequence of training tokens, the model attempts to map each n-gram (sequence of n consecutive words) to the set of words that commonly follow it.
Generation starts from a seed n-gram and produces random yet somewhat structurally plausible sequences by sampling from learned transitions.
    - `train(tokens)`: Attempts to build the n-gram to next-word map using a sliding window
    - `generate(seed, length)`: Attempts to generate a sequence of up to `length` tokens by sampling from the transition map
    - Attempts to internally use `join()` and `split()` to manage token sequence transformations

In this module, I attempted to make the component useful for generating creative and/or non-sensical text, exploring token dependencies, and prototyping bot responses and/or poetry.

#### Potential Improvements??

- **Seed Smoothing**: If a seed is unseen, fallback to random selection from the full model keyspace to avoid early generation halts.
- **Frequency-Aware Sampling**: Currently uniform sampling over possible next words; could maybe weight next word probabilities by frequency.
- **Export / Import Model**: Consider maybe saving the model map as JSON for future reuse and/or debugging.
- **Higher-Order Example**: Demonstrate bigram and/or trigram generation in `example_pipeline.cpp` to showcase structural fluency.
- **Model Size Frequency**: Add a method to return the number of unique n-gram keys in the model (`model_size()`).



## Installation and Setup

### Local Installation (Development Mode)
To install WeirdNLPs locally (e.g., my own PC):
```bash
pip install .
```

Please ensure that you have all necessary dependencies installed beforehand. I suppose you can try `pip install -e .` to install in "editable" mode, which can hopefully reflect code changes immediately without needing to reinstall.

#### Important:
This project does depend on [LibTorch](https://pytorch.org/get-started/locally/)...
Please download the appropriate version for your OS, unzip it, and place the `libtorch/` folder at the root of this repository:
```text
WeirdNLPs/
├── libtorch/
│   ├── include/
│   └── lib/
├── src/
├── CMakeLists.txt
```

If you happen to use pretrained models and/or artifacts, make sure a `dist/` folder is present. I excluded it from GitHub so that the repo is more lightweight.

### Installing through PyPI
When I (hopefully) publish this to places such as PyPI, one will hopefully be able to install WeirdNLPs like so:
```bash
pip install weirdnlp
```
Stay tuned for more updates!

### Prerequisites

Prior to installing WeirdNLPs, please ensure you have the following:

- **Python ≥ 3.8**: For `pip` installation and Python-side use
- **C++17-Compatible Compiler**: For instance `g++`, `clang++`, etc. for C++ extensions
- **pybind11**: I tried to include this via the `scikit-build-core` part in the `pyproject.toml` file...presumably one can install this manually too..
- **Torch / LibTorch**: Needed for Phase 11 regarding Deep Learning Models, not required for most classical NLP tasks
- **CMake ≥ 3.15**: Required by `scikit-build-core` for project configuration
- **`make`**: For building and testing C++ modules

If installing from source, please ensure the aforementioned tools are accessible via your system path.

#### Required Resource Files

**WeirdNLPs** does depend on many `.txt` resource files (e.g., `english_vocab.txt`, `ner_lexicon.txt`, etc.) for its core functionalities such as lemmatization, POS tagging, sentiment analysis, and named entity recognition (NER).

Such files are **not hardcoded** into the compiled library, so one may choose to check the following:

##### Please ensure that the files are available. They should be one or a combo of these:
- **Automatically installed with the package**: *(hopefully planned for future releases)*, **or** 
- **Downloaded manually** from the `data/` folder in this repository, **or**
- **Passed explicitly** into any module that expects a lexicon file via file path

> **Missing files may cause some features (such as NER or POS tagging) to fail** with file read errors.

---

#### Common Resource Files

| **Purpose**               | **File**                  |
|---------------------------|---------------------------|
| Lemmatizer                | `english_vocab.txt`       |
| POS Tagger                | `pos_lexicon.txt`         |
| Named Entity Lexicon      | `ner_lexicon.txt`         |
| Sentiment Lexicon         | `sentiment_lexicon.txt`   |
| Embedding Support         | `mini_glove.txt`          |

---

If you install via `pip` and happen to not see such files, then try cloning the repository and copy the necessary `.txt` files into your working directory or an appropriate path.

#### For Developers and Contributors
To clone and install locally: 
```bash
git clone https://github.com/eyang72004/WeirdNLPs.git
cd WeirdNLPs
pip install .
```

#### For End Users (also covered in Installing through PyPI in Installation and Setup somewhat...)
```bash
pip install weirdnlp
```


## How to Use

Once you have installed the library, you can attempt to import it into your Python project:
```python
import weirdnlp
```

Use modules such as (but are not limited to) `Corpus`, `Lemmatizer`, etc. from `weirdnlp`. I tried to make all 13 phases accessible from Python via pybind11 bindings.

## Development Goals

This library was created for several personal and experimental purposes:
- To continue applying my self-studies in C++
- To experiment with `pybind11` for Python bindings
- To test and see if I am able to build an NLP library from scratch, making weird decisions and all in the process of doing so
- To see if it is possible to bridge classical NLP with modern machine/deep learning workflows
- To embrace imperfection, weirdness, and even confusion all as parts of the learning process (no perfect LLMs here...)

In principle, I was not trying to strive for a production-ready, state-of-the-art library -- I simply wanted to try everything, "break some rules", and document some of the chaos if I encountered any along the way (which is apparently a lot...).


## Known Quirks

Quite frankly, there are quite a bit of weird (and somewhat janky) quirks lying in WeirdNLPs. Here are a few:
- Some modules (like NER) rely primarily on fragile lexicons and naive matching.
- Unicode support is limited in several places (e.g., tokenization, vectorization...).
- I do not perceive any GPU support yet....though Phase 11 does use CPU-based TorchScript inference..
- Many files seem to lack full unit tests and edge case coverage (though I still am trying to figure such matters out..).
- POS tagger may label anything as NN if the word is not known.
- Everything assumes English as of now.
- As far as this goes, the Deep Learning wrapper likely assumes one prepares Torch inputs and tokenizes things appropriately.

In a nutshell, everything is still quite experimental -- so while it may serve as a (hopefully) useful learning tool and/or prototype, I am not entirely sure if it will be ready for production environments, critical applications, and/or formal academic use.


## Future Works

So in principle -- WeirdNLPs is never meant to be perfect -- however, it *was* meant to grow.

While many of the future works could be addressed through the potential improvements for each of the 13 phases, there is a lot more to explore...
- **Multilingual NLP Support**: Especially for tokenization, stemming, and lemmatization. English-only pipelines can be limiting, and expanding language support would make this toolkit far more versatile.
- **Integrated Neural Tokenizers**: Possibly bringing in Byte Pair Encoding or SentencePiece could help bridge the gap between traditional NLP and modern transformer-based preprocessing.
- **Interactive CLI / REPL Tools**: Allowing users to run tokenization, vectorization, tagging, or even sentiment analysis from the terminal would make WeirdNLPs (hopefully) easier and more fun to interact with.
- **Unit Testing Coverage**: Every phase deserves robust test suites. I think that using `pytest` on the Python side and maybe `catch2` and/or `doctest` for C++ would hopefully ensure things break less often...or at least more predictably..
- **Better Error Reporting**: Logging unknown words, token mismatches, or fallback behavior would help both users and developers understand what the system is doing (or failing to do).
- **Auto-Downloads and Sanity Checks**: Theoretically speaking, downloading lexicons and/or models with one command. One may not have to manually copy as much anymore -- just better defaults, versioning, and hopefully clean installs.
- **Modular CMake Options**: As of now, everything seems to build at once. Future versions could offer phase-specific flags so users only install what they need.
- **Publishing to PyPI (hopefully for real)**: `pip install weirdnlp` will hopefully get to experience full publishing with wheels, version tags, and CI workflows in scope.
- **Training Routines for ML / DL Phases**: Allow users to train Naive Bayes and/or Logistic Regression on their own data, or even fine-tune a TorchScript model from Phase 11. Model tuning belongs here.

---

Honestly, WeirdNLPs is meant to be a platform for **learning, tinkering, and hacking** (not in an unethical sense..) -- not necessarily a drop-in production solution. But, I think, that is the beauty of things...it is weird, yet educational and (hopefully) useful in a sense.

Pull requests and contributions are welcome.
