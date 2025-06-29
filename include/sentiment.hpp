#pragma once
#ifndef LIBRARY_NLP_SENTIMENT_HPP
#define LIBRARY_NLP_SENTIMENT_HPP

#include <string>
#include <vector>
#include <unordered_map>

namespace weirdnlp {
    
    // SentimentAnalyzer performs lexicon-based sentiment scoring of tokens
    class SentimentAnalyzer {
    public:

        // Constructor: loads the sentiment lexicon from the file
        explicit SentimentAnalyzer(const std::string& lexicon_path);

        // Computes the total sentiment score for a vector of tokens
        int score(const std::vector<std::string>& tokens) const;

        // Classifies the total score into a label: Positive, Negative, or Neutral
        std::string classify(int score) const;

    private:
        std::unordered_map<std::string, int> sentiment_map; // Word-to-Score Map

        // Loads the lexicon file (each line must be: word score)
        void load_lexicon(const std::string& path);
        
    };
}





#endif // LIBRARY_NLP_SENTIMENT_HPP