#include "corpus.hpp"
#include "tokenization.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <iostream>
#include <regex>

namespace weirdnlp {

    // Constructor loads raw text and then normalizes it
    Corpus::Corpus(const std::string& filepath) {
        raw_text = load_file(filepath);
        normalized_text = normalize(raw_text);
    }

    // Reads file content into a string
    std::string Corpus::load_file(const std::string& filepath) const {
        std::ifstream file(filepath);
        if (!file) {
            std::cerr << "Error opening file: " << filepath << "\n";
            return "";
        }

        std::ostringstream buffer;
        buffer << file.rdbuf(); // Load entire file contents into buffer
        return buffer.str();
    }

    // Normalizes text by lowercasing and replacing punctuation with spaces
    std::string Corpus::normalize(const std::string& text) const {
        std::string norm;
        for (size_t i = 0; i < text.size(); i++) {
            char ch = text[i];

            if (std::isalnum(ch) || std::isspace(ch)) {
                norm += std::tolower(ch);
            } else if (ch == '-' && i > 0 && i < text.size() - 1 && std::isalpha(text[i - 1]) && std::isalpha(text[i + 1])) {
                norm += ch; // I will try here to keep the hyphen in between two letters
            } else if (ch == '\'' && i > 0 && i < text.size() - 1 && std::isalpha(text[i - 1]) && std::isalpha(text[i + 1])) {
                norm += ch; // Apostrophe case
            }

            

        }
        
        
        
        /*
        for (char ch : text) {
            if (std::isalnum(ch) || std::isspace(ch)) {
                norm += std::tolower(ch); // Convert letters to lowercase 
            } else {
                norm += ' '; // Replace punctuation with space
            }
        }
            */
        return norm;
    }

    // Getters for raw original text
    std::string Corpus::get_raw_text() const {
        return raw_text;
    }   

    // Getters for normalized text
    std::string Corpus::get_normalized_text() const {
        return normalized_text;
    }

    // Uses sentence splitting from tokenization module
    std::vector<std::string> Corpus::split_sentences() const {
        return weirdnlp::sentence_split(raw_text);
    }

    // Uses regex-based tokenizer on normalized text
    std::vector<std::string> Corpus::tokenize_words() const {
        return weirdnlp::regex_tokenize(normalized_text);
    }
}