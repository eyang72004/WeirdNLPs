#include "embeddings.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>


namespace weirdnlp {

    // Loads embeddings from file. Assumes format: word val1 val2 ...
    // Each line corresponds to a word and its vector representation
    EmbeddingModel::EmbeddingModel(const std::string& filepath) {
        std::ifstream file(filepath);

        if(!file) {
            std::cerr << "Error opening file: " << filepath << "\n";
            return;
        }

        std::string line;

        // Read each line from the embeddings file
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word;
            iss >> word; // Extract the word (first token on the line)

            std::vector<float> vec;
            float value;

            // Read the embedding values
            while (iss >> value) {
                vec.push_back(value);
            }


            // Set the expected dimension on the first successful vector
            if (dim == 0) {
                dim = vec.size(); // Initialize dimension on first line
            }

            // Only keep the embedding if it has the correct dimension
            if ((int)vec.size() == dim) {
                embeddings[word] = vec;
            } else {
                std::cerr << "Dimension mismatch for word: " << word << "\n";
            }
        }
    }

    // Check whether a word has a vector representation in the model
    bool EmbeddingModel::contains(const std::string& word) const {
        return embeddings.find(word) != embeddings.end();
    }

    // Retrieve the vector for a given word
    // If the word is not found, return a zero vector of correct dimension
    std::vector<float> EmbeddingModel::get_vector(const std::string& word) const {
        auto it = embeddings.find(word);
        return (it != embeddings.end()) ? it->second : std::vector<float>(dim, 0.0f);
    }

    // Compute dot product between two vectors
    float EmbeddingModel::dot(const std::vector<float>& a, const std::vector<float>& b) const {
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Compute L2 (Euclidean) norm of a vector
    float EmbeddingModel::norm(const std::vector<float>& a) const {
        return std::sqrt(dot(a, a));
    }

    // Compute cosine similarity between two words using their embeddings
    // Tried to use formula: cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    float EmbeddingModel::cosine_similarity(const std::string& word1, const std::string& word2) const {
        auto vec1 = get_vector(word1);
        auto vec2 = get_vector(word2);

        return dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10f); // Adding a small epsilon to avoid division by zero
    }

    // Perform a word analogy operation:
    // word1 : word2 :: word3 : ?
    
    std::vector<float> EmbeddingModel::analogy(const std::string& word1, const std::string& word2, const std::string& word3) const {
        // Computes vec(word2) - vec(word1) + vec(word3)
        auto vec1 = get_vector(word1);
        auto vec2 = get_vector(word2);
        auto vec3 = get_vector(word3);

        std::vector<float> result(dim, 0.0f);
        for (int i = 0; i < dim; ++i) {
            result[i] = vec2[i] - vec1[i] + vec3[i];
        }
        return result;
    }
}