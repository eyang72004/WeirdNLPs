#pragma once
#ifndef LIBRARY_NLP_EMBEDDINGS_HPP
#define LIBRARY_NLP_EMBEDDINGS_HPP

#include <string>
#include <vector>
#include <unordered_map>

namespace weirdnlp {
    
    // Represents static word embedding model, e.g., Word2Vec, GloVe, FastText.
    class EmbeddingModel {
    public:

        // Load pre-trained embeddings from a file
        explicit EmbeddingModel(const std::string& filepath);
        
        // Checks whether a word exists in embedding dictionary
        bool contains(const std::string& word) const;

        // Returns embedding vector for a given word
        std::vector<float> get_vector(const std::string& word) const;

        // Computes cosine similarity between two word vectors
        float cosine_similarity(const std::string& word1, const std::string& word2) const;

        //Computes analogy: word2 - word1 + word3
        std::vector<float> analogy(const std::string& word1, const std::string& word2, const std::string& word3) const;




        
    private:
        std::unordered_map<std::string, std::vector<float>> embeddings; // word -> vector
        int dim = 0; // embedding dimension

        // Helper: computes dot product of two vectors
        float dot(const std::vector<float>& a, const std::vector<float>& b) const;

        // Helper: computes L2 norm (magnitude) of a vector
        float norm(const std::vector<float>& a) const;    
    };
}



#endif // LIBRARY_NLP_EMBEDDINGS_HPP