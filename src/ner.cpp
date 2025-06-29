#include "ner.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>
#include <iostream>

namespace weirdnlp {
    
    // Normalize by lowercasing and replacing underscores with spaces
    std::string NERTagger::normalize(const std::string& word) {
        std::string result;
        for (char c : word) {
            
            if (std::isalnum(c) || c == '-' || c == ' ') {       
                result += std::tolower(static_cast<unsigned char>(c));
            } 
            else if (c == '_') {
                result += ' ';
            } else if (c == ' ') {
                result += ' ';
            }

            
        }
        return result;
    }

    // Upper Case thingamajigs??
    std::string to_title_case(const std::string& word) {
        if (word.empty()) {
            return word;
        }
        std::string result = word;
         bool capitalize_next = true;

        
        for (size_t i = 0; i < result.size(); i++) {
            if (capitalize_next && std::isalpha(result[i])) {
                result[i] = std::toupper(static_cast<unsigned char>(result[i]));
                capitalize_next = false;

                // Heuristic/brute force for I think one apostrophe case
                if (i > 0 && result[i - 1] == '\'' && (i + 1 >= result.size() || !std::isalpha(result[i + 1]))) {
                    result[i] = std::tolower(static_cast<unsigned char>(result[i]));
                }
            } else {
                result[i] = std::tolower(static_cast<unsigned char>(result[i]));
            }

            if (result[i] == '-' || result[i] == '\'') {
                capitalize_next = true;
            }
        }
        
        /*
        result[0] = std::toupper(static_cast<unsigned char>(result[0]));
        for (size_t i = 1; i < result.size(); i++) {
            result[i] = std::tolower(static_cast<unsigned char>(result[i]));
        }
            */
        return result;
    }





    // Loads a lexicon mapping entities (e.g., "Obama") to their types (e.g., "PERSON")
    void NERTagger::load_lexicon(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error opening lexicon file: " << path << "\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string entity, tag;
            if (iss >> entity >> tag) {
                entity_map[normalize(entity)] = tag;
            }
        }
    }


    // Constructor: initialize the NERTagger with a given lexicon path
    NERTagger::NERTagger(const std::string& lexicon_path) {
        load_lexicon(lexicon_path);
    }

    // Heuristic fallback for tagging unknown words:
    // - If 4 digits: assume it is a year (e.g., DATE)
    // - If capitalized: assume it might be a proper noun (POSSIBLE_NAME)
    // - Else: tag as '0' (outside any named entity)
    std::string NERTagger::heuristic_tag(const std::string& word) const {
        if (std::regex_match(word, std::regex(R"(\d{4})"))) return "DATE";
        if (std::regex_match(word, std::regex(R"([A-Z][a-z]+)"))) return "POSSIBLE_NAME";
        return "O"; // Outside (Default)
    }

    // Tags a list of tokens using the lexicon if available,
    // otherwise uses simple regex-based heuristics
    std::vector<std::pair<std::string, std::string>> NERTagger::tag(const std::vector<std::string>& tokens) const {
        std::vector<std::pair<std::string, std::string>> tagged;
        size_t i = 0;

        while (i < tokens.size()) {
            bool found = false;

            // Attempted to try trigram, then bigram, then unigram
            for (int n = 4; n >= 1; --n) {
                if (i + n <= tokens.size()) {
                    std::string phrase;
                    for (int j = 0; j < n; j++) {
                        phrase += tokens[i + j];
                        if (j < n - 1) phrase += " ";
                    }

                    std::string norm_phrase = normalize(phrase);
                    auto it = entity_map.find(norm_phrase);
                    if (it != entity_map.end()) {
                        std::string tag = it->second;
                        for (int j = 0; j < n; j++) {
                            tagged.emplace_back(to_title_case(tokens[i + j]), tag);
                        }
                        i += n;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                std::string tag = heuristic_tag(tokens[i]);
                tagged.emplace_back(to_title_case(tokens[i]), tag);
                ++i;
            }
        }
        
        
        
        
        
        
        
        
        
        
        
     
        return tagged;
    }
}