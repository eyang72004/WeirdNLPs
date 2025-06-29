#include "../include/vectorization.hpp"
#include <iostream>

int main() {

    // A toy corpus of tokenized documents
    std::vector<std::vector<std::string>> corpus = {
        {"the", "cat", "sat"},
        {"the", "dog", "sat"},
        {"the", "dog", "barked"}
    };

    weirdnlp::Vocabulary vocab;

    vocab.build(corpus); // Build vocabulary from corpus


    // Create a BoW vectorizer and compute vectors
    weirdnlp::BoWVectorizer bow(vocab);

    // Create and fit a TF-IDF vectorizer
    weirdnlp::TFIDFVectorizer tfidf;
    tfidf.fit(corpus);

    std::cout << "Bag of Words Vectorization:\n";
    for (const auto& doc : corpus) {
        for (int count : bow.vectorize(doc)) {
            std::cout << count << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nTF-IDF Vectorization:\n";
    for (const auto& doc : corpus) {
        for (double score : tfidf.transform(doc)) {
            std::cout << score << " ";
        }
        std::cout << "\n";
    }

    return 0;
}