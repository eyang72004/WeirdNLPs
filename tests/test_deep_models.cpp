#include "../include/deep_models.hpp"
#include <iostream>

int main() {
    
    // Load TorchScript model from file
    weirdnlp::TorchScriptModel model("../data/example_model.pt");


    // Sample token IDs (pretending this is like BERT's [CLS]...This is a test[SEP])
    std::vector<int64_t> token_ids = {101, 2023, 2003, 1037, 2742, 102}; // Example input IDs


    // Run inference and print raw output logits
    auto output = model.infer(token_ids);

    std::cout << "Logits:\n";
    for (float val : output) {
        std::cout << val << " ";
    }

    std::cout << "\n";

    // Get the predicted class (argmax of logits)
    int predicted = model.classify(token_ids);
    std::cout << "Predicted class index: " << predicted << "\n";

    return 0;

}