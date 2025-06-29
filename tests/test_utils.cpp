#include "../include/utils.hpp"
#include <iostream>

int main() {
    std::string text = "Hello, World! NLP is fun.";

    // Test case: Convert to lowercase
    std::cout << "Original: " << text << "\n";
    std::cout << "Lowercase: " << weirdnlp::to_lower(text) << "\n";

    // Test case: remove punctuation
    std::cout << "Without punctuation: " << weirdnlp::remove_punctuation(text) << "\n";

    // Test case: cosine similarity for two identical vectors
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {1.0, 2.0, 3.0};
    std::cout << "Cosine Similarity: " << weirdnlp::cosine_similarity(vec1, vec2) << "\n";

    return 0;
}