#include "../include/corpus.hpp"
#include <iostream>

int main() {

    // Load and process corpus from a file
    weirdnlp::Corpus corpus("../data/sample_text.txt");

    // Show raw text (as-is from file)
    std::cout << "Raw Text:\n" << corpus.get_raw_text() << "\n\n";

    // Show normalized text (lowercase, no punctuation)
    std::cout << "Normalized Text:\n" << corpus.get_normalized_text() << "\n\n";


    // Sentence splitting on raw text
    std::cout << "\nSentences:\n";

    for (const auto& sentence : corpus.split_sentences()) {
        std::cout << "- " << sentence << "\n";
    }


    // Word tokenization on normalized text
    std::cout << "\nWords:\n";
    for (const auto& word : corpus.tokenize_words()) {
        std::cout << word << "\n";
    }
    std::cout << "\n";

    return 0;
}