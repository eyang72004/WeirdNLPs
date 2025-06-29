import weirdnlp
import os

# Resolve all paths
base_path = os.path.abspath(os.path.dirname(__file__))
vocab_path = os.path.join(base_path, "data", "english_vocab.txt")
text_path = os.path.join(base_path, "data", "sample_text.txt")
embed_path = os.path.join(base_path, "data", "mini_glove.txt")
pos_path = os.path.join(base_path, "data", "pos_lexicon.txt")
ner_path = os.path.join(base_path, "data", "ner_lexicon.txt")
sent_path = os.path.join(base_path, "data", "sentiment_lexicon.txt")

print("\n=== Porter Stemmer ===")
print("porter_stem('reviving') =", weirdnlp.porter_stem("reviving"))

print("\n=== Lemmatizer ===")
lemmatizer = weirdnlp.Lemmatizer(vocab_path)
print("lemmatize('running') =", lemmatizer.lemmatize("running"))

print("\n=== Corpus ===")
corpus = weirdnlp.Corpus(text_path)
print("corpus raw text:", corpus.get_raw_text().strip())
print("tokenized:", corpus.tokenize_words())

print("\n=== Vocabulary & BOW ===")
vocab = weirdnlp.Vocabulary()
docs = [["apple", "banana", "apple"], ["banana", "orange"]]
vocab.build(docs)
print("vocab size:", vocab.size())
bow = weirdnlp.BowVectorizer(vocab)
print("BOW vector for ['apple', 'orange']:", bow.vectorize(["apple", "orange"]))

print("\n=== TF-IDF ===")
tfidf = weirdnlp.TFIDFVectorizer()
tfidf.fit(docs)
print("TF-IDF for ['apple', 'banana']:", tfidf.transform(["apple", "banana"]))

print("\n=== Embeddings ===")
try:
    embed_model = weirdnlp.EmbeddingModel(embed_path)
    print("cosine_similarity('dog', 'puppy'):", embed_model.cosine_similarity("dog", "puppy"))
except Exception as e:
    print("Embeddings error:", e)

print("\n=== POS Tagging ===")
try:
    pos = weirdnlp.POSTagger(pos_path)
    print("POS tags:", pos.tag(["He", "sings", "well"]))
except Exception as e:
    print("POS error:", e)

print("\n=== NER Tagging ===")
try:
    ner = weirdnlp.NERTagger(ner_path)
    print("NER tags:", ner.tag(["Elon", "Musk", "founded", "SpaceX"]))
except Exception as e:
    print("NER error:", e)

print("\n=== Sentiment Analysis ===")
try:
    sentiment = weirdnlp.SentimentAnalyzer(sent_path)
    tokens = ["This", "movie", "was", "amazing"]
    score = sentiment.score(tokens)
    print("Sentiment score:", score)
    print("Sentiment label:", sentiment.classify(score))
except Exception as e:
    print("Sentiment error:", e)
