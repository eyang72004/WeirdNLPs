#include "sentiment.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace weirdnlp {
    
    // Reads a sentiment lexicon from file and stores it in the map
    // File format: one word per line with its sentiment score (e.g., " love 2")
    void SentimentAnalyzer::load_lexicon(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error opening lexicon file: " << path << "\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word;
            int score;
            if (iss >> word >> score) {
                sentiment_map[word] = score;
            }
        }
    }

    // Constructor: calls load_lexicon to initialize the sentiment map
    SentimentAnalyzer::SentimentAnalyzer(const std::string& lexicon_path) {
        load_lexicon(lexicon_path);
    }


    // Computes the sentiment score of a list of tokens by summing up individual word scores
    int SentimentAnalyzer::score(const std::vector<std::string>& tokens) const {
        int total = 0;
        for (const auto& word : tokens) {
            auto it = sentiment_map.find(word);
            if (it != sentiment_map.end()) {
                total += it->second;
            }
        }
        return total;
    }

    // Converts raw sentiment score into a lable: Positive, Negative, or Neutral
    std::string SentimentAnalyzer::classify(int score) const {
        if (score > 0) {
            return "Positive";
        } else if (score < 0) {
            return "Negative";
        } else {
            return "Neutral";
        }
    }
}