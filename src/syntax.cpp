#include "syntax.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

namespace weirdnlp {
    
    // Loads the POS lexicon into an unordered_map
    // Each line in the file should contain <word> <POS-tag>
    void POSTagger::load_lexicon(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error opening lexicon file: " << path << "\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word, tag;
            if (iss >> word >> tag) {
                pos_map[word] = tag;
            }
        }

    }

    // Constructor that loads the lexicon file during initialization
    POSTagger::POSTagger(const std::string& lexicon_path) {
        load_lexicon(lexicon_path);
    }

    // If a word is not found in the lexicon, this method applies heuristic rules
    std::string POSTagger::heuristic_tag(const std::string& word) const {
        if (std::regex_match(word, std::regex(".*ing$"))) return "VBG"; // Gerund
        if (std::regex_match(word, std::regex(".*ed$"))) return "VBD"; // Past tense
        if (std::regex_match(word, std::regex(".*s$"))) return "NNS"; // Plural noun
        if (std::regex_match(word, std::regex(".*ly$"))) return "RB"; // Adverb
        return "NN"; // Default to noun
    }

    // Tags each token using the lexicon if available, otherwise applies heuristic rules
    std::vector<std::pair<std::string, std::string>> POSTagger::tag(const std::vector<std::string>& tokens) const {
        std::vector<std::pair<std::string, std::string>> tagged;
        for (const auto& word : tokens) {
            auto it = pos_map.find(word);
            std::string tag = (it != pos_map.end()) ? it->second : heuristic_tag(word);
            tagged.emplace_back(word, tag);
        }
        return tagged;
    }
}