#include "../include/tokenization.hpp"
#include "../include/stemming.hpp"
#include "../include/lemmatization.hpp"
#include "../include/corpus.hpp"
#include "../include/vectorization.hpp"
#include "../include/embeddings.hpp"
#include "../include/syntax.hpp"
#include "../include/ner.hpp"
#include "../include/sentiment.hpp"
#include "../include/ml_models.hpp"
#include "../include/deep_models.hpp"
#include "../include/markov_chain.hpp"
#include <iostream>

int main() {
    
    // Phase 4: Loading and preprocessing the corpus hopefully
    weirdnlp::Corpus corpus("../data/sample_text.txt");
    weirdnlp::Lemmatizer lemmatizer("../data/english_vocab.txt");

    auto corpus_tokens = corpus.tokenize_words();
    std::vector<std::vector<std::string>> docs = { corpus_tokens };

    // Corpus
    std::cout << "\n==== Final Token Stream (Lemmatized Corpus) ====\n";
    for (const auto& word : corpus_tokens) {
        std::cout << lemmatizer.lemmatize(word) << " ";
    }
    std::cout << "\n";

    // Phase 2-3: Lemmatization on New Text Hopefully
    std::string text1 = "He was running and eating while the cars went by.";
    auto tokens1 = weirdnlp::regex_tokenize(text1);
    std::cout << "\n==== Lemmatized Tokens ====\n";
    for (const auto& token : tokens1) {
        std::cout << lemmatizer.lemmatize(token) << " ";
    }
    std::cout << "\n\n";

    // Phase 2: Stemming Hopefully (Need to expand on this later)
    std::string text2 = "The programmers agreed to keep motoring and singing in the rain.";
    auto tokens2 = weirdnlp::regex_tokenize(text2);
    std::cout << "\n==== Stemmed Tokens ====\n";
    for (const auto& token : tokens2) {
        std::cout << weirdnlp::porter_stem(token) << " ";
    }
    std::cout << "\n\n";

    // Phase 1: Tokenization display Hopefully
    std::cout << "\n==== Tokens (Regex) ====\n";
    for (const auto& token : tokens2) {
        std::cout << token << " ";
    }
    std::cout << "\n";

    auto sentences = weirdnlp::sentence_split(text2);
    std::cout << "\n==== Sentence Splitting ====\n";
    for (const auto& sentence : sentences) {
        std::cout << sentence << "\n";
    }

    // Phase 5: Vectorization Hopefully
    weirdnlp::Vocabulary vocab;
    vocab.build(docs);

    weirdnlp::BoWVectorizer bow(vocab); 
    weirdnlp::TFIDFVectorizer tfidf;
    tfidf.fit(docs);

    auto bow_vector = bow.vectorize(corpus_tokens);
    auto tfidf_vector = tfidf.transform(corpus_tokens);

    std::cout << "\n==== Bag of Words Vectorization ====\n";
    for (auto val : bow_vector) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    std::cout << "\n==== TF-IDF Vectorization ====\n";
    for (auto val : tfidf_vector) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Phase 6: Embeddings Hopefully
    weirdnlp::EmbeddingModel model("../data/mini_glove.txt");
    std::cout << "\n==== Embedding Similarity ====\n";
    std::cout << "Cosine Similarity between 'king' and 'queen': " << model.cosine_similarity("king", "queen") << "\n";
    std::cout << "Cosine Similarity between 'dog' and 'cat': " << model.cosine_similarity("dog", "cat") << "\n";
    std::cout << "apple vs car : " << model.cosine_similarity("apple", "car") << "\n";

    // Phase 7: Syntax Hopefully
    weirdnlp::POSTagger tagger("../data/pos_lexicon.txt");

    std::cout << "\n==== POS Tagging ====\n";
    for (const auto& [word, tag] : tagger.tag(corpus_tokens)) {
        std::cout << word << " / " << tag << "\n";
    }
    std::cout << "\n";

    // Phase 8: NER Hopefully
    weirdnlp::NERTagger ner("../data/ner_lexicon.txt");

    std::cout << "\n==== NER Results ====\n";
    for (const auto& [word, tag] : ner.tag(corpus_tokens)) {
        std::cout << word << " / " << tag << "\n";
    }
    std::cout << "\n";

    // Phase 9: Sentiment Analysis Hopefully
    weirdnlp::SentimentAnalyzer sentiment_analyzer("../data/sentiment_lexicon.txt");

    int score = sentiment_analyzer.score(corpus_tokens);
    std::string label = sentiment_analyzer.classify(score);

    std::cout << "\n==== Sentiment Analysis ====\n";
    std::cout << "Sentiment Score: " << score << "\n";
    std::cout << "Classification: " << label << "\n";
    std::cout << "\n";

    // Phase 10: ML Models Hopefully
    std::vector<std::vector<std::string>> train_docs = {
        {"good", "movie"},
        {"bad", "movie"},
        {"great", "film"},
        {"terrible", "film"},
    };

    std::vector<std::string> nb_labels = {
        "positive",
        "negative",
        "positive",
        "negative",
    };

    std::vector<int> lr_labels = {1, 0, 1, 0};


    weirdnlp::Vocabulary clf_vocab;
    clf_vocab.build(train_docs);

    weirdnlp::BoWVectorizer clf_bow(clf_vocab);
    std::vector<std::vector<int>> clf_vectors;
    for (const auto& doc : train_docs) {
        clf_vectors.push_back(clf_bow.vectorize(doc));
    }

    weirdnlp::NaiveBayesClassifier nb;
    nb.fit(clf_vectors, nb_labels);
    std::cout << "\n==== Naive Bayes Prediction ====\n";
    std::cout << "Prediction: " << nb.predict(clf_bow.vectorize({"great", "movie"})) << "\n";


    weirdnlp::LogisticRegressionClassifier lr;
    lr.fit(clf_vectors, lr_labels);
    std::cout << "\n==== Logistic Regression Prediction ====\n";
    std::cout << "Prediction: " << lr.predict(clf_bow.vectorize({"bad", "film"})) << "\n";

    // Phase 11: Deep Learning Models Hopefully
    weirdnlp::TorchScriptModel torch_model("../data/example_model.pt");
    std::vector<int64_t> token_ids = {101, 2023, 2003, 1037, 2742, 102}; 

    auto logits = torch_model.infer(token_ids);
    std::cout << "\n==== TorchScript Model Inference ====\n";
    std::cout << "Logits:\n";

    for (float val : logits) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    int pred = torch_model.classify(token_ids);
    std::cout << "Predicted Class Index: " << pred << "\n";

    // Phase 13: Markov Chains Hopefully....
    std::string input_text = "language models are cool and language models can generate text";
    std::vector<std::string> tokens = weirdnlp::regex_tokenize(input_text);

    weirdnlp::MarkovChain markov(1); // Unigram
    markov.train(tokens);

    std::cout << "\n=== [EXAMPLE PIPELINE] Markov Chain Text Generation ===\n" << std::endl;
    std::vector<std::string> output = markov.generate("language", 12);

    for (const auto& word : output) {
        std::cout << word << " ";
    }
    std::cout << "\n";

    return 0;
}
