#pragma once
#ifndef LIBRARY_NLP_MARKOV_CHAIN_HPP
#define LIBRARY_NLP_MARKOV_CHAIN_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <random>
#include <sstream>

namespace weirdnlp {
    class MarkovChain {
    public:
        explicit MarkovChain(int n = 2);

        void train(const std::vector<std::string>& tokens);
        std::vector<std::string> generate(const std::string& seed, int length = 20) const;

    private:
        int ngram_size;
        std::unordered_map<std::string, std::vector<std::string>> model;
        mutable std::mt19937 rng;

        std::vector<std::string> split(const std::string& text) const;
        std::string join(const std::vector<std::string>& tokens, int start, int n) const;
    };
}


#endif // LIBRARY_NLP_MARKOV_CHAIN_HPP