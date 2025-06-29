#include "tokenization.hpp"
#include <sstream>
#include <regex>
#include <iterator>


namespace weirdnlp {

    // Splits input string at spaces and returns all words (includes punctuation)
    std::vector<std::string> whitespace_tokenize(const std::string& text) {
        std::istringstream iss(text);
        return {std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>{}};
    }

    // Uses regex to find word tokens (alphanumeric sequences). Removes punctuation.
    std::vector<std::string> regex_tokenize(const std::string& text) {
        //std::regex word_regex(R"([\w'-]+)");
        std::regex word_regex(R"(\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b)"); // Supposed to match sequences of word characters between word boundaries...
        std::sregex_iterator begin(text.begin(), text.end(), word_regex), end;
        std::vector<std::string> tokens;
        for (auto i = begin; i != end; ++i) {
            tokens.push_back(i->str()); // Add each matched word
        }
        return tokens;
    }

    // Uses regex to split the input into complete sentences ending with punctuation (., !, ?)
    std::vector<std::string> sentence_split(const std::string& text) {
        std::regex sentence_regex(R"([^.!?]+[.!?])"); // Match sequences ending in sentence punctuation
        std::sregex_iterator begin(text.begin(), text.end(), sentence_regex), end;
        std::vector<std::string> sentences;
        for (auto i = begin; i != end; ++i) {
            sentences.push_back(i->str()); // Add each sentence match
        }
        return sentences;
    }
}