#include "lemmatization.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace weirdnlp {

    // Constructor calls load_dictionary to initialize the lemmatization map.
    Lemmatizer::Lemmatizer(const std::string& dict_path) {
        load_dictionary(dict_path);
    }

    // Reads inflected and lemma word pairs from a file and populates the map.
    void Lemmatizer::load_dictionary(const std::string& path) {
        std::ifstream infile(path);
        if (!infile) {
            std::cerr << "Error opening dictionary file: " << path << "\n";
            return;
        }

        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string inflected, lemma;

            // Each line should contain a pair: inflected form and lemma form
            if (iss >> inflected >> lemma) {
                lemmatization_map[inflected] = lemma;
            }
        }
    }

    // Looks up the lemma form for the input word, or returns the original word if not found.
    std::string Lemmatizer::lemmatize(const std::string& word) const {
        auto it = lemmatization_map.find(word);
        return it != lemmatization_map.end() ? it->second : word; // Return the original word if no lemma is found
    }
}