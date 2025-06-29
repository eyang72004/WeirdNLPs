#pragma once
#ifndef LIBRARY_NLP_LEMMATIZATION_HPP
#define LIBRARY_NLP_LEMMATIZATION_HPP

#include <string>
#include <unordered_map>

namespace weirdnlp {

    // Lemmatizer class maps inflected forms to their base forms (lemmas).
    class Lemmatizer {
    public:

        // Constructor: loads a dictionary from the specified file path.
        explicit Lemmatizer(const std::string& dict_path);

        // Returns the lemma for the input word, or the original word if no lemma is found.
        std::string lemmatize(const std::string& word) const;

    private:

        // A hash map to store inflected -> lemma mappings.
        std::unordered_map<std::string, std::string> lemmatization_map;

        // Loads inflected-lemma pairs from a file into the lemmatization_map.
        void load_dictionary(const std::string& path);
    
    };

}




#endif // LIBRARY_NLP_LEMMATIZATION_HPP