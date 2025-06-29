#include "../include/sentiment.hpp"
#include "../include/tokenization.hpp"
#include <iostream>

int main() {
    
    // Input sentence containing both positive and negative sentiments
    std::string text = "I love programming, but I hate bugs.";

    // Tokenize the sentence into individual words
    auto tokens = weirdnlp::regex_tokenize(text);

    // Initialize sentiment analyzer with a predefined lexicon file
    weirdnlp::SentimentAnalyzer sentiment_analyzer("../data/sentiment_lexicon.txt");


    // Calculate the total sentiment score and classify it
    int score = sentiment_analyzer.score(tokens);
    std::string label = sentiment_analyzer.classify(score);

    // Print the results
    std::cout << "Sentiment Score: " << score << "\n";  
    std::cout << "Classification: " << label << "\n";

    return 0;
}