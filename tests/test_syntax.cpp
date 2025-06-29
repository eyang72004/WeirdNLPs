#include "../include/syntax.hpp"
#include "../include/tokenization.hpp"
#include <iostream>

int main() {
    
    // Example input sentence
    std::string text = "The quick brown fox jumps over the lazy dog.";


    // Step 1: Tokenize the input using regex-based word tokenizer
    auto tokens = weirdnlp::regex_tokenize(text);

    // Step 2: Initialize the POS tagger with a lexicon file
    weirdnlp::POSTagger tagger("../data/pos_lexicon.txt");


    // Step 3: Tag each token with a part-of-speech label
    auto tagged = tagger.tag(tokens);

    // Step 4: Display the results
    std::cout << "POS Tagging Results:\n";
    for (const auto& [word, tag] : tagged) {
        std::cout << word << " / " << tag << "\n";
    }
    std::cout << "\n";

    return 0;
}