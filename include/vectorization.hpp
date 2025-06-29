#pragma once 
#ifndef LIBRARY_NLP_VECTORIZATION_HPP
#define LIBRARY_NLP_VECTORIZATION_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace weirdnlp {

    // Builds a vocabulary from a corpus and maps words to indices.
    class Vocabulary {
    public:

        // Creates vocabulary from list of tokenized documents
        void build(const std::vector<std::vector<std::string>>& documents);

        // Returns index of given word, or -1 if not found
        int get_index(const std::string& word) const;

        // Returns size of vocabulary
        size_t size() const;

        // Accessor for underlying map
        const std::unordered_map<std::string, int>& get_vocab() const;
    
    private:
        std::unordered_map<std::string, int> word_to_index;
    };

    // Creates a bag-of-words vector from a document
    class BoWVectorizer {
    public:
        explicit BoWVectorizer(const Vocabulary& vocab);

        // Transforms a document into a bag-of-words vector
        std::vector<int> vectorize(const std::vector<std::string>& document) const;

    private:
        const Vocabulary& vocab;
    };

    // Computes TF-IDF vectors from corpus and applies to documents
    class TFIDFVectorizer {
    public:

        // Computes document frequency stats across corpus
        void fit(const std::vector<std::vector<std::string>>& corpus);

        // Converts a document into TF-IDF vector
        std::vector<double> transform(const std::vector<std::string>& document) const;

    private:
        Vocabulary vocab;
        std::vector<int> document_frequencies;
        int total_documents = 0;
    };
}

#endif // LIBRARY_NLP_VECTORIZATION_HPP