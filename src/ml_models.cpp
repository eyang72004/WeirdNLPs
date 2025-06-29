#include "ml_models.hpp"
#include <cmath>
#include <numeric>
#include <iostream>

namespace weirdnlp {
    // -------- Naive Bayes Classifier --------
    void NaiveBayesClassifier::fit(const std::vector<std::vector<int>>& X, const std::vector<std::string>& y) {
        std::unordered_map<std::string, int> class_counts;
        std::unordered_map<std::string, std::vector<int>> word_counts;

        vocab_size = X[0].size(); // Assume uniform feature size

        // Count words and documents per class
        for (size_t i = 0; i < X.size(); ++i) {
            const std::string& cls = y[i];
            class_counts[cls]++;
            if (word_counts[cls].empty()) word_counts[cls] = std::vector<int>(vocab_size, 0);
            
            
            for (size_t j = 0; j < vocab_size; ++j) {
                word_counts[cls][j] += X[i][j];
            }
        }

        // Compute log prior probabilities for each class
        int total_docs = X.size();
        for (const auto& [cls, count] : class_counts) {
            class_probs[cls] = std::log((double)count / total_docs);
        }

        // Ok...so I tried to compute log likelihoods with Laplace smoothing..as they call it maybe....
        for (const auto& [cls, counts] : word_counts) {
            int total_words = std::accumulate(counts.begin(), counts.end(), 0);
            std::vector<double> probs(vocab_size);
            

            for (size_t j = 0; j < vocab_size; ++j) {
                probs[j] = std::log((counts[j] + 1.0) / (total_words + vocab_size));
            }
            word_probs[cls] = probs;
        }
    }

    std::string NaiveBayesClassifier::predict(const std::vector<int>& x) const {
        std::string best_class;
        double best_log_prob = -INFINITY;


        // Evaluate log posterior for each class
        for (const auto& [cls, log_prior] : class_probs) {
            double log_prob = log_prior;
            for (int j = 0; j < vocab_size; ++j) {
                log_prob += x[j] * word_probs.at(cls)[j];
            }
            if (log_prob > best_log_prob) {
                best_log_prob = log_prob;
                best_class = cls;
            }
        }

        return best_class;
    }

    // -------- Logistic Regression Classifier --------
    double LogisticRegressionClassifier::sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

    void LogisticRegressionClassifier::fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y, double learning_rate, int epochs) {
        int m = X.size(); // Number of training examples
        int n = X[0].size(); // Number of features
        weights = std::vector<double>(n, 0.0); // Initialize weights
        

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> gradients(n, 0.0);
            double bias_grad = 0.0;
            
            for (int i = 0; i < m; ++i) {
                // Compute prediction: sigmoid(w⋅x + b)
                double z = std::inner_product(weights.begin(), weights.end(), X[i].begin(), 0.0) + bias;
                double pred = sigmoid(z);
                double error = pred - y[i];

                // Accumulate gradients 
                for (int j = 0; j < n; ++j) {
                gradients[j] += error * X[i][j];
                }
                bias_grad += error;
            }

            // Apply weight and bias updates
            for (int j = 0; j < n; ++j) {
                weights[j] -= learning_rate * gradients[j] / m;
            }

            bias -= learning_rate * bias_grad / m;
            
            
            
            
        }
    }

    int LogisticRegressionClassifier::predict(const std::vector<int>& x) const {
        double z = std::inner_product(weights.begin(), weights.end(), x.begin(), 0.0) + bias;
        return sigmoid(z) > 0.5 ? 1 : 0;
    }
}