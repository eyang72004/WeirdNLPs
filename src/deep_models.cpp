#include "deep_models.hpp"
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>



namespace weirdnlp {

    // Load a serialized TorchScript model (.pt file)
    TorchScriptModel::TorchScriptModel(const std::string& model_path) {
        try {
            model = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << "\n";
            throw;
        }
    }

    // Performs forward pass on the model and returns logits as float vector
    std::vector<float> TorchScriptModel::infer(const std::vector<int64_t>& input_ids) {
        
        // Convert input_ids into a single batch (1 x N tensor)
        torch::Tensor input_tensor = torch::tensor(input_ids, torch::dtype(torch::kFloat)).unsqueeze(0);

        // Prepare inputs in TorchScript-compatible format
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Forward pass 
        at::Tensor output = model.forward(inputs).toTensor();

        // Extract logits from tensor to std::vector<float>
        std::vector<float> result(output.size(1));
        for (int i = 0; i < output.size(1); ++i) {
            result[i] = output[0][i].item<float>();
        }
        return result;
    }

    // Classifies input by selecting the index with the highest logit
    int TorchScriptModel::classify(const std::vector<int64_t>& input_ids) {
        auto logits = infer(input_ids);
        int best_index = 0;
        float best_value = logits[0];

        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_value) {
                best_value = logits[i];
                best_index = i;
            }
        }
        return best_index;
    }
}