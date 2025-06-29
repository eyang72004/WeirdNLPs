#include "../include/ml_models.hpp"
#include "../include/vectorization.hpp"
#include <iostream>

int main() {

    // Sample labeled dataset (binary sentiment classification)
    std::vector<std::vector<std::string>> docs = {
        {"good", "movie"},
        {"bad", "movie"},
        {"great", "film"},
        {"terrible", "film"},
    };

    std::vector<std::string> labels = {
        "positive",
        "negative",
        "positive",
        "negative",
    };

    std::vector<int> binary_labels= {1, 0, 1, 0}; // For Logistic Regression

    // Build vocabulary
    weirdnlp::Vocabulary vocab;
    vocab.build(docs);

    // Convert to BoW vectors
    weirdnlp::BoWVectorizer bow(vocab);
    std::vector<std::vector<int>> X;
    for (const auto& doc : docs) {
        X.push_back(bow.vectorize(doc));
    }

    // ---- Naive Bayes ----
    weirdnlp::NaiveBayesClassifier nb;
    nb.fit(X, labels);
    std::cout << "Naive Bayes Prediction: " << nb.predict(bow.vectorize({"great", "movie"})) << "\n";



    // ---- Logistic Regression ----
    weirdnlp::LogisticRegressionClassifier lr;
    lr.fit(X, binary_labels);
    std::cout << "Logistic Regression Prediction: " << lr.predict(bow.vectorize({"bad", "film"})) << "\n";



    // ---- Just a janky attempt at TF-IDF Vectorization...One could even say it is just for illustration here ----
    weirdnlp::TFIDFVectorizer tfidf;
    tfidf.fit(docs);

    std::vector<std::vector<double>> X_tfidf;
    for (const auto& doc : docs) {
        X_tfidf.push_back(tfidf.transform(doc));
    }

    std::cout << "TF-IDF Vectorization:\n";
    for (const auto& vec : X_tfidf) {
        for (double val : vec) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }


    
    return 0;
}