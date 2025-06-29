#include "vectorization.hpp"
#include <iostream>

namespace weirdnlp {
    
    // Vocabulary

    // Builds a mapping from each unique word to a unique index
    // Input: Documents - a list of tokenized documents
    void Vocabulary::build(const std::vector<std::vector<std::string>>& documents) {
        word_to_index.clear(); // Reset previous state
        int index = 0;

        // For each document...
        for (const auto& doc : documents) {

            // For each word in the document...
            for (const auto& word : doc) {

                // If this word has not been seen before, assign it a new index
                if (word_to_index.find(word) == word_to_index.end()) {
                    word_to_index[word] = index++;
                }
            }
        }
    }
    
    // Retrieve index of word in the vocabulary
    // Returns index of word, or -1 if not found
    int Vocabulary::get_index(const std::string& word) const {
        auto it = word_to_index.find(word);
        return it != word_to_index.end() ? it->second : -1; 
    }
    
    // Return number of words in the vocabulary
    size_t Vocabulary::size() const {
        return word_to_index.size();
    }

    // Accessor for the entire vocabulary map
    const std::unordered_map<std::string, int>& Vocabulary::get_vocab() const {
        return word_to_index;
    }

    // BoW Vectorizer

    // Constructor: initialize with an already built vocabulary
    BoWVectorizer::BoWVectorizer(const Vocabulary& vocab) : vocab(vocab) {}

    // Vectorize a single document using the bag-of-words model
    // Output: vector of word counts corresponding to vocab indices
    std::vector<int> BoWVectorizer::vectorize(const std::vector<std::string>& document) const {
        std::vector<int> vec(vocab.size(), 0); // Initialize frequency vector

        // Count each word in the document
        for(const auto& word : document) {
            int index = vocab.get_index(word);
            if (index != -1) {
                vec[index]++; // Increment count at the word's index
            }
        }
        return vec;
    }

    
    // TF-IDF Vectorizer

    // Fit TF-IDF vectorizer to a corpus
    // Computes document frequency (DF) for each word
    void TFIDFVectorizer::fit(const std::vector<std::vector<std::string>>& corpus) {
        vocab.build(corpus); // Build vocabulary from the corpus
        total_documents = corpus.size(); // Store total doc count
        document_frequencies = std::vector<int>(vocab.size(), 0); // Init DF array

        // For each document, keep track of which words appear
        for (const auto& doc : corpus) {
            std::unordered_map<int, bool> seen;
            for (const auto& word : doc) {
                int index = vocab.get_index(word);
                if (index != -1 && !seen[index]) {
                    document_frequencies[index]++;
                    seen[index] = true; // Avoid counting duplicates in the same doc
                }
            }
        }
    }

    // Transform a document into its TF-IDF vector representation
    std::vector<double> TFIDFVectorizer::transform(const std::vector<std::string>& document) const {
        std::vector<double> tfidf(vocab.size(), 0.0); // Result vector
        std::vector<int> term_counts(vocab.size(), 0); // Count of word appearances

        // Step 1: Count occurrences of each word
        for (const auto& word : document) {
            int index = vocab.get_index(word);
            if (index != -1) {
                term_counts[index]++;
            }
        }

        // Step 2: Applying what I reckon to be the TF-IDF formula
        for (size_t i = 0; i< tfidf.size(); ++i) {
            if (term_counts[i]  > 0 && document_frequencies[i] > 0) {
                double tf = term_counts[i]; // Raw term frequency
                double idf = std::log((1.0 + total_documents) / (1.0 + document_frequencies[i])) + 1.0;
                tfidf[i] = tf * idf;
            }
        }
        return tfidf;
    }
   
}