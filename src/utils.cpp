#include "utils.hpp"
#include <cctype>
#include <sstream>
#include <numeric>
#include <cmath>

namespace weirdnlp {

    // Converts each character in the input string to lowercase
    std::string to_lower(const std::string& text) {
        std::string out;
        for (char ch : text) {
            out += std::tolower(ch);
        }
        return out;
    }

    // Removes all non-alphanumeric characters (except spaces) from the input string
    std::string remove_punctuation(const std::string& text) {
        std::string out;
        for (char ch : text) {
            if (std::isalnum(ch) || std::isspace(ch)) {
                out += ch;
            }
        }
        return out;
    }

    // Splits a space-separated string into individual tokens (words)
    std::vector<std::string> split_by_space(const std::string& text) {
        std::istringstream iss(text);
        return {std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>{}};
    }

    // Computes cosine similarity between two vectors:
    // (dot(a, b)) / (||a|| * ||b||)
    double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10); // Adding a small value to avoid division by zero
    }
}