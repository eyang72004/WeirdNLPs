#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/stemming.hpp"
#include "../include/lemmatization.hpp"
#include "../include/corpus.hpp"
#include "../include/vectorization.hpp"
#include "../include/embeddings.hpp"
#include "../include/syntax.hpp"
#include "../include/ner.hpp"
#include "../include/sentiment.hpp"
#include "../include/ml_models.hpp"
#include "../include/deep_models.hpp"
#include "../include/utils.hpp"
#include "../include/markov_chain.hpp"

namespace py = pybind11;

PYBIND11_MODULE(weirdnlp, m) {
    m.doc() = "Python bindings for WeirdNLP C++ library...hopefully they work somehow";

    // Bind stemming function hopefully.....
    m.def("porter_stem", &weirdnlp::porter_stem, "Apply Porter stemming algorithm hopefully....");

    // Bind the lemmatizer class hopefully......
    py::class_<weirdnlp::Lemmatizer>(m, "Lemmatizer") 
        .def(py::init<const std::string&>())
        .def("lemmatize", &weirdnlp::Lemmatizer::lemmatize);

    // Corpus Loading Hopefully...
    py::class_<weirdnlp::Corpus>(m, "Corpus")
        .def(py::init<const std::string&>())
        .def("get_raw_text", &weirdnlp::Corpus::get_raw_text)
        .def("get_normalized_text", &weirdnlp::Corpus::get_normalized_text)
        .def("split_sentences", &weirdnlp::Corpus::split_sentences)
        .def("tokenize_words", &weirdnlp::Corpus::tokenize_words);

    // Vocabulary and Vectorization Loading Hopefully...
    py::class_<weirdnlp::Vocabulary>(m, "Vocabulary")
        .def(py::init<>())
        .def("build", &weirdnlp::Vocabulary::build)
        .def("get_index", &weirdnlp::Vocabulary::get_index)
        .def("get_vocab", &weirdnlp::Vocabulary::get_vocab)
        .def("size", &weirdnlp::Vocabulary::size);

    py::class_<weirdnlp::BoWVectorizer>(m, "BowVectorizer")
        .def(py::init<const weirdnlp::Vocabulary&>())
        .def("vectorize", &weirdnlp::BoWVectorizer::vectorize);

    py::class_<weirdnlp::TFIDFVectorizer>(m, "TFIDFVectorizer")
        .def(py::init<>())
        .def("fit", &weirdnlp::TFIDFVectorizer::fit)
        .def("transform", &weirdnlp::TFIDFVectorizer::transform);

    // Embeddings Loading hopefully.....
    py::class_<weirdnlp::EmbeddingModel>(m, "EmbeddingModel")
        .def(py::init<const std::string&>())
        .def("contains", &weirdnlp::EmbeddingModel::contains)
        .def("get_vector", &weirdnlp::EmbeddingModel::get_vector)
        .def("cosine_similarity", &weirdnlp::EmbeddingModel::cosine_similarity)
        .def("analogy", &weirdnlp::EmbeddingModel::analogy);

    // POS Tagging and Loading Hopefully.....
    py::class_<weirdnlp::POSTagger>(m, "POSTagger")
        .def(py::init<const std::string&>())
        .def("tag", &weirdnlp::POSTagger::tag);

    // NER Loading hopefully......
    py::class_<weirdnlp::NERTagger>(m, "NERTagger")
        .def(py::init<const std::string&>())
        .def("tag", &weirdnlp::NERTagger::tag);

    // Sentiment Analysis Loading hopefully....
    py::class_<weirdnlp::SentimentAnalyzer>(m, "SentimentAnalyzer")
        .def(py::init<const std::string&>())
        .def("score", &weirdnlp::SentimentAnalyzer::score)
        .def("classify", &weirdnlp::SentimentAnalyzer::classify);


    // Naive Bayes Classifier Loading Hopefully....
    py::class_<weirdnlp::NaiveBayesClassifier>(m, "NaiveBayesClassifier")
        .def(py::init<>())
        .def("fit", &weirdnlp::NaiveBayesClassifier::fit)
        .def("predict", &weirdnlp::NaiveBayesClassifier::predict);

    // Logistic Regression Classifier Binding Hopefully....
    py::class_<weirdnlp::LogisticRegressionClassifier>(m, "LogisticRegressionClassifier")
        .def(py::init<>())
        .def("fit", &weirdnlp::LogisticRegressionClassifier::fit,
                py::arg("X"), py::arg("y"), py::arg("learning_rate") = 0.1, py::arg("epochs") = 1000)
        .def("predict", &weirdnlp::LogisticRegressionClassifier::predict);

    // Deep Models Binding Hopefully....
    py::class_<weirdnlp::TorchScriptModel>(m, "TorchScriptModel")
            .def(py::init<const std::string&>())
            .def("infer", &weirdnlp::TorchScriptModel::infer)
            .def("classify", &weirdnlp::TorchScriptModel::classify);

    // Markov Chain Binding Hopefully...
    py::class_<weirdnlp::MarkovChain>(m, "MarkovChain")
            .def(py::init<int>(), py::arg("n") = 2)
            .def("train", &weirdnlp::MarkovChain::train)
            .def("generate", &weirdnlp::MarkovChain::generate,
                  py::arg("seed"), py::arg("length") = 20);

}