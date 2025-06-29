#include "../include/stemming.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<std::string> words = {
        "caresses", "ponies", "ties", "caress", "cats", "agreed",
    "plastered", "bled", "motoring", "sing", "hopping", "sky", "programmer", "different", "rational", "skies", "skiing", "influencer",
    "dying", "abilities", "happily", "happiness", "happy", "programming", "coding", "boxes", 
    "running", "beautiful", "beautifulness"
    };

    std::cout << "Stemming results:\n";
    

    // Print each word and its stemmed version
    for (const auto& word : words) {
        std::cout << word << " -> " << weirdnlp::porter_stem(word) << "\n";
    }

    return 0;
}