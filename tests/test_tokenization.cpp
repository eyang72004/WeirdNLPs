#include "../include/tokenization.hpp"
#include <iostream>

int main() {
    std::string text = "Hello World! This is a test. Let us try tokenization.";


    // Test 1: Split on whitespace (simple sentence-based split, punctuation preserved)
    auto whitespace = weirdnlp::whitespace_tokenize(text);
    std::cout << "Whitespace Tokenization:\n";
    for (const auto& token : whitespace) {
        std::cout << token << "\n";
    }


    // Test 2: Use regex to extract only alphanumeric word tokens (punctuation removed)
    auto regex = weirdnlp::regex_tokenize(text);
    std::cout << "\nRegex Tokenization:\n";
    for (const auto& token : regex) {
        std::cout << token << "\n";
    }


    // Test 3: Split text into sentences using punctuation markers
    auto sentences = weirdnlp::sentence_split(text);
    std::cout << "\nSentence Splitting:\n";
    for (const auto& sentence : sentences) {
        std::cout << sentence << "\n";
    }

    return 0;
}