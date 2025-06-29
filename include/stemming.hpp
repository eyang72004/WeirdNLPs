#pragma once
#ifndef LIBRARY_NLP_STEMMING_HPP
#define LIBRARY_NLP_STEMMING_HPP

#include <string>

namespace weirdnlp {

    // Applies the Porter stemming algorithm to reduce a word to its stem/root form.
    std::string porter_stem(const std::string& word);
}



#endif // LIBRARY_NLP_STEMMING_HPP