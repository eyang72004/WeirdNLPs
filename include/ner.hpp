#pragma once
#ifndef LIBRARY_NLP_NER_HPP
#define LIBRARY_NLP_NER_HPP

#include <string>
#include <vector>
#include <unordered_map>

namespace weirdnlp {

    // NERTagger performs Named Entity Recognition using a lookup table and heuristic rules.
    class NERTagger {
    public:

        // Constructor: loads lexicon mapping entities to their labels (e.g., PERSON, ORGANIZATION)
        explicit NERTagger(const std::string& lexicon_path);

        // Tags each token with a named entity label or default heuristic tag
        std::vector<std::pair<std::string, std::string>> tag(const std::vector<std::string>& tokens) const;

    private:
        std::unordered_map<std::string, std::string> entity_map; // Stores known entities and their NER tags

        // Loads lexicon from file: each line must be formatted as "<entity> <NER-tag>""
        void load_lexicon(const std::string& path);

        // Applies basic regex-based tagging for unknown entities
        std::string heuristic_tag(const std::string& word) const;

        static std::string normalize(const std::string& word); // Lowercase + Underscore-to-space
    };
}



#endif // LIBRARY_NLP_NER_HPP