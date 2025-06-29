#pragma once
#ifndef WEIRD_NLP_UTILS_HPP
#define WEIRD_NLP_UTILS_HPP

#include <string>
#include <vector>

namespace weirdnlp {

    // Converts all characters in the input string to lowercase
    std::string to_lower(const std::string& text);

    // Removes punctuation characters from input string, preserving alphanumeric and space characters
    std::string remove_punctuation(const std::string& text);

    // Splits input string by whitespace into a vector of tokens
    std::vector<std::string> split_by_space(const std::string& text);

    // Computes cosine similarity between two vectors of the same dimension
    double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b);
}



#endif // WEIRD_NLP_UTILS_HPP