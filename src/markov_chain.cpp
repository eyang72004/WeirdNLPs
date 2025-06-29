#include "markov_chain.hpp"

using namespace weirdnlp;

// Constructor for MarkovChain class
// Initializes n-gram size and seeds the random number generator
MarkovChain::MarkovChain(int n) : ngram_size(n), rng(std::random_device{}()) {}

// Train the Markov model with a sequence of tokens
void MarkovChain::train(const std::vector<std::string>& tokens) {

    // Not enough tokens to build any n-gram
    if (tokens.size() <= ngram_size) return;

    // Slide a window of size "n-gram_size" over the tokens
    // and map each n-gram to the next word that follows it
    for (size_t i = 0; i + ngram_size < tokens.size(); i++) {
        std::string key = join(tokens, i, ngram_size); // Current n-gram key
        model[key].push_back(tokens[i + ngram_size]); // Add next word to the model
    }
}

// Generate a sequence of words starting from a seed n-gram
std::vector<std::string> MarkovChain::generate(const std::string& seed, int length) const {
    std::vector<std::string> result;
    std::string current = seed;
    
    result.push_back(current); // Start with the seed

    for (int i = 0; i < length; i++) {
        auto it = model.find(current); // Look up next word options
        if (it == model.end()) break; // Stop if n-gram not found

        const std::vector<std::string>& next_words = it->second;
        std::uniform_int_distribution<size_t> dist(0, next_words.size() - 1);
        std::string next = next_words[dist(rng)]; // Randomly pick next word
        result.push_back(next);

        // Update current context by shifting window forward
        std::vector<std::string> context = split(current);
        context.erase(context.begin()); // Remove first word
        context.push_back(next); // Add new word at end
        current = join(context, 0, ngram_size); // Recreate n-gram key
    }

    return result;
}

// Split string (like n-gram) into tokens using space
std::vector<std::string> MarkovChain::split(const std::string& text) const {
    std::stringstream ss (text);
    std::string word;
    std::vector<std::string> tokens;
    while (ss >> word) tokens.push_back(word);
    return tokens;
}

// Join a sequence of of tokens from tokens[start] for n tokens into a single string
std::string MarkovChain::join(const std::vector<std::string>& tokens, int start, int n) const {
    std::stringstream ss;
    for (int i = 0; i < n; i++) {
        if (i > 0) ss << " ";
        ss << tokens[start + i];
    }
    return ss.str();
}