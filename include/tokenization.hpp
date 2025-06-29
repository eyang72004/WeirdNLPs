#pragma once
#ifndef LIBRARY_NLP_TOKENIZATION_HPP
#define LIBRARY_NLP_TOKENIZATION_HPP

#include <string>
#include <vector>

namespace weirdnlp {

    // Tokenizers

    // Splits input text into tokens based on whitespace (e.g., "Hello World!" -> ["Hello", "World!"])
    std::vector<std::string> whitespace_tokenize(const std::string &text);

    // Uses regex to split input text into word-only tokens (punctuation removed)
    std::vector<std::string> regex_tokenize(const std::string &text);

    // Splits input text into sentences based on punctuation (., !, ?) using regex
    std::vector<std::string> sentence_split(const std::string &text);
}


#endif // LIBRARY_NLP_TOKENIZATION_HPP