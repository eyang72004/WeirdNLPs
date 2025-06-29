#include "../include/markov_chain.hpp"
#include <iostream>

int main() {
    std::vector<std::string> tokens = {
        "the", "cat", "sat", "on", "the", "mat",
        "the", "cat", "slept", "on", "the", "sofa"
    };

    weirdnlp::MarkovChain chain(1); // Unigram Model Hopefully...
    chain.train(tokens);

    std::cout << "=== [TEST] Markov Chain Generation ===" << std::endl;
    std::vector<std::string> generated = chain.generate("the", 20);

    for (const auto& word : generated) {
        std::cout << word << " ";
    }

    std::cout << std::endl;

    return 0;
}