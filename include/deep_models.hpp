#pragma once
#ifndef LIBRARY_NLP_DEEP_MODELS_HPP
#define LIBRARY_NLP_DEEP_MODELS_HPP

#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

namespace weirdnlp {

    // Ok so....this is the first time I tried to use PyTorch, and hopefully I did so by wrapping around a PyTorch TorchScript 
    // model for inference.
    class TorchScriptModel {
    public:

        // Constructor loads a pre-trained TorchScript model from disk
        explicit TorchScriptModel(const std::string& model_path);

        // Runs inference on input token IDs and returns raw logits
        std::vector<float> infer(const std::vector<int64_t>& input_ids);


        // Returns index of highest scoring class from the logits
        int classify(const std::vector<int64_t>& input_ids);


    private:
        torch::jit::script::Module model; // TorchScript model loaded from file

    };
}

#endif // LIBRARY_NLP_DEEP_MODELS_HPP