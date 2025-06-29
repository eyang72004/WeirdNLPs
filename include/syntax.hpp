#pragma once
#ifndef LIBRARY_NLP_SYNTAX_HPP
#define LIBRARY_NLP_SYNTAX_HPP

#include <string>
#include <vector>
#include <unordered_map>

namespace weirdnlp {

    // Simple rule-based POS tagger using a lexicon and heuristic rules
    class POSTagger {
    public:
        // Constructor: loads a lexicon file containing word-tag pairs
        explicit POSTagger(const std::string& lexicon_path);

        // Tags a list of tokens with their corresponding POS tags
        std::vector<std::pair<std::string, std::string>> tag(const std::vector<std::string>& tokens) const;


    private:
        std::unordered_map<std::string, std::string> pos_map; // word -> POS tag

        // Loads word-tag mappings from given lexicon file
        void load_lexicon(const std::string& path);

        // Applies rule-based heuristics when a word is not found in the lexicon
        std::string heuristic_tag(const std::string& word) const;
    };

}


#endif // LIBRARY_NLP_SYNTAX_HPP