#include "../include/lemmatization.hpp"
#include <iostream>
#include <vector>

int main() {
    // Initialize the lemmatizer with the path to the dictionary file
    weirdnlp::Lemmatizer lemmatizer("../data/english_vocab.txt");

    // Sample words to lemmatize
    std::vector<std::string> words = {
        "running", "eats", "feet", "cars", "went", "is", "singing"
    };

    // Apply lemmatization and display each mapping
    std::cout << "Lemmatization results:\n";
    for (const auto& word : words) {
        std::cout << word << " -> " << lemmatizer.lemmatize(word) << "\n";
    }

    return 0;
}