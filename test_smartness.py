import weirdnlp

def test_stemming():
    print("[Stemming] 'relational' →", weirdnlp.porter_stem("relational"))

def test_lemmatization():
    lemmatizer = weirdnlp.Lemmatizer("data/english_vocab.txt")
    print("[Lemmatization] 'went' →", lemmatizer.lemmatize("went"))

def test_corpus():
    corpus = weirdnlp.Corpus("data/sample_text.txt")
    print("[Corpus] Sentences →", corpus.split_sentences())
    print("[Corpus] Tokens →", corpus.tokenize_words())

def test_tfidf():
    tfidf = weirdnlp.TFIDFVectorizer()
    tfidf.fit([["dog", "barks"], ["cat", "meows"]])
    print("[TF-IDF] for ['dog', 'barks'] →", tfidf.transform(["dog", "barks"]))

def test_pos():
    tagger = weirdnlp.POSTagger("data/pos_lexicon.txt")
    print("[POS] ['I', 'run'] →", tagger.tag(["I", "run"]))

def test_ner():
    ner = weirdnlp.NERTagger("data/ner_lexicon.txt")
    print("[NER] ['Barack', 'Obama', 'visited', 'Paris'] →", ner.tag(["Barack", "Obama", "visited", "Paris"]))

def test_sentiment():
    analyzer = weirdnlp.SentimentAnalyzer("data/sentiment_lexicon.txt")
    tokens = ["I", "hate", "this", "movie"]
    score = analyzer.score(tokens)
    print(f"[Sentiment] Score: {score}, Label: {analyzer.classify(score)}")

def test_markov():
    mc = weirdnlp.MarkovChain(n=1)
    mc.train(["the", "cat", "sat"])
    print("[Markov Chain] Seed 'the' →", mc.generate("the", length=3))

if __name__ == "__main__":
    test_stemming()
    test_lemmatization()
    test_corpus()
    test_tfidf()
    test_pos()
    test_ner()
    test_sentiment()
    test_markov()
