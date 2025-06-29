#include "../include/ner.hpp"
#include "../include/tokenization.hpp"
#include <iostream>

int main() {

    // Sample input sentence for NER.
    std::string text = "Barack Obama didn't visit the new NASA in New York on July 4 with LeBron James and Chris Callison-Burch and Mark O'Connor. Mark his words.";

    // Tokenize the sentence using regex (splits by words)
    auto tokens = weirdnlp::regex_tokenize(text);

    // Create NER tagger using predefined lexicon file
    weirdnlp::NERTagger ner("../data/ner_lexicon.txt");


    // Tag each token and print named entity classification
    std::cout << "NER Results:\n";
    for (const auto& [word, tag] : ner.tag(tokens)) {
        std::cout << word << " / " << tag << "\n";
    }

    std::cout << "\n";

    return 0;
}