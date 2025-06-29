import weirdnlp


import os

base_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data") if os.path.exists(os.path.join(base_path, "data")) else os.path.join(base_path, "..", "..", "data")

vocab_path = os.path.join(data_path, "english_vocab.txt")
text_path = os.path.join(data_path, "sample_text.txt")
glove_path = os.path.join(data_path, "mini_glove.txt")
pos_path = os.path.join(data_path, "pos_lexicon.txt")
ner_path = os.path.join(data_path, "ner_lexicon.txt")
sentiment_path = os.path.join(data_path, "sentiment_lexicon.txt")

print("=== [TEST] Porter Stemmer ===")
print("porter_stem('caresses') =", weirdnlp.porter_stem("caresses"))

print("\\n=== [TEST] Lemmatizer ===")
lemmatizer = weirdnlp.Lemmatizer(vocab_path)
print("lemmatize('cars') =", lemmatizer.lemmatize("cars"))

print("\\n=== [TEST] Corpus ===")
corpus = weirdnlp.Corpus(text_path)
print("corpus raw text:", corpus.get_raw_text())

print("\\n=== [TEST] Vocabulary & BOW ===")
vocab = weirdnlp.Vocabulary()
vocab.build([["the", "dog", "barked"], ["the", "cat", "meowed"]])
print("vocab size:", vocab.size())
bow = weirdnlp.BowVectorizer(vocab)
print("BOW vector for ['the', 'dog']:", bow.vectorize(["the", "dog"]))

print("\\n=== [TEST] TF-IDF ===")
tfidf = weirdnlp.TFIDFVectorizer()
tfidf.fit([["the", "dog"], ["dog", "barked"]])
print("TF-IDF for ['the', 'dog']:", tfidf.transform(["the", "dog"]))

print("\\n=== [TEST] Embeddings ===")
embedding_model = weirdnlp.EmbeddingModel(glove_path)
print("cosine_similarity('king', 'queen'):", embedding_model.cosine_similarity("king", "queen"))

print("\\n=== [TEST] POS Tagging ===")
pos_tagger = weirdnlp.POSTagger(pos_path)
print("POS tags:", pos_tagger.tag(["The", "cat", "runs"]))

print("\\n=== [TEST] NER Tagging ===")
ner_tagger = weirdnlp.NERTagger(ner_path)
print("NER tags:", ner_tagger.tag(["Obama", "visited", "New_York"]))

print("\\n=== [TEST] Sentiment Analysis ===")
sentiment = weirdnlp.SentimentAnalyzer(sentiment_path)
tokens = ["I", "love", "this", "film"]
score = sentiment.score(tokens)
print("Sentiment score:", score)
print("Sentiment label:", sentiment.classify(score))