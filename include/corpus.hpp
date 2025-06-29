#pragma once
#ifndef LIBRARY_NLP_CORPUS_HPP
#define LIBRARY_NLP_CORPUS_HPP

#include <string>
#include <vector>

namespace weirdnlp {

    // Corpus class handles loading, normalizing, and tokenizing text data from file.
    class Corpus {
    public:
        
        // Constructor: loads and preprocesses the text from a file.
        explicit Corpus(const std::string& filepath);

        // Returns original raw text
        std::string get_raw_text() const;

        // Returns normalized text (lowercased, no punctuation)
        std::string get_normalized_text() const;


        // Splits the raw text into sentences
        std::vector<std::string> split_sentences() const;

        // Tokenizes normalized text into individual words using regex
        std::vector<std::string> tokenize_words() const;

    private:
        std::string raw_text;
        std::string normalized_text;

        // Reads content from the file
        std::string load_file(const std::string& filepath) const;

        // Normalizes the text (lowercase, remove punctuation)
        std::string normalize(const std::string& text) const;
        
    };
}


#endif // LIBRARY_NLP_CORPUS_HPP