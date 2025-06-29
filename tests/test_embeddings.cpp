#include "../include/embeddings.hpp"
#include <iostream>

int main() {

    // Load mini GloVe file (ensure the path is correct)
    weirdnlp::EmbeddingModel model("../data/mini_glove.txt");

    // Print similarity between common analogical pairs
    std::cout << "Cosine Similarity between 'king' and 'queen': " << model.cosine_similarity("king", "queen") << "\n";
    std::cout << "Cosine Similarity between 'man' and 'woman': " << model.cosine_similarity("man", "woman") << "\n";


    // Print vector for a single word
    std::cout << "Vector for 'apple': ";
    for (auto val : model.get_vector("apple")) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Vector result of analogy: king - man + woman = ?
    auto analogy_vec = model.analogy("man", "king", "woman");
    std::cout << "\nVector for analogy (man::king : woman::?): ";
    for (auto val : analogy_vec) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}