#pragma once
#ifndef LIBRARY_NLP_ML_MODELS_HPP
#define LIBRARY_NLP_ML_MODELS_HPP

#include <string>
#include <vector>
#include <unordered_map>

namespace weirdnlp {

    // Naive-Bayes Classifier for text classification using multinomial likelihood
    class NaiveBayesClassifier {
    public:

        // Trains model on bag-of-words features (X) and corresponding class labels (y)
        void fit(const std::vector<std::vector<int>>& X, const std::vector<std::string>& y);

        // Predicts most likely class for a new BoW feature vector x
        std::string predict(const std::vector<int>& x) const;

    private:
        std::unordered_map<std::string, double> class_probs; // Prior log-probabilities of classes
        std::unordered_map<std::string, std::vector<double>> word_probs; // Likelihoods of words given in a class
        int vocab_size = 0;
    };

    // Logistic Regression Classifier for binary classification
    class LogisticRegressionClassifier {
    public:

        // Trains the model using gradient descent on binary labels
        void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y, double learning_rate = 0.1, int epochs = 1000);

        // Predicts a binary label (0 or 1) for input feature vector x
        int predict(const std::vector<int>& x) const;

    private:
        std::vector<double> weights; // Linear model weights
        double bias = 0.0; // Interception
        double sigmoid(double z) const; // Activation function
    };
}



#endif // LIBRARY_NLP_ML_MODELS_HPP
