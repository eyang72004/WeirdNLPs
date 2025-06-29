#include "stemming.hpp"
#include <cctype>
#include <unordered_map>


namespace weirdnlp {
    
    // Returns true if the character at index 'i' in 'word' is a consonant
    static bool is_consonant(const std::string& word, int index) {
        char ch = word[index];
        if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') {
            return false;
        }

        // Special Case: 'y' is a consonant if the previous letter is a vowel
        if (ch == 'y') return (index == 0) ? true : !is_consonant(word, index - 1);
        return true;
    }

    static bool ends_with_double_consonant(const std::string& word) {
        if (word.size() < 2) {
            return false;
        }
        return word[word.size() - 1] == word[word.size() - 2] && is_consonant(word, word.size() - 1);
    }

    // Measures the number of VC (vowel-consonant) sequences in the word
    static int measure(const std::string& word) {
       int i = 0;
       int n = 0;

       int len = word.size();
       while (i < len && is_consonant(word, i)) {
        i++; // Skip initial consonants hopefully
       }
       while (i < len) {
        while (i < len && !is_consonant(word, i)) i++; // Skip vowels hopefully
        while (i < len && is_consonant(word, i)) i++; // Then consonants?
        n++;
       }
       /*
        int count = 0;
        bool prev_c = false;
        for (size_t i = 0; i < word.size(); i++) {
            bool curr_c = is_consonant(word, i);
            if (i > 0 && prev_c && !curr_c) count++;
            prev_c = curr_c;
        }
        */
        return n;
    }

    

    // Returns true if 'word' contains a vowel
    static bool contains_vowel(const std::string& word) {
        for (size_t i = 0; i < word.size(); ++i) {
            if (!is_consonant(word, i)) {
                return true;
            }
        }
        return false;
    }


    // Check if 'word' ends with the suffix
    static bool ends_with(const std::string& word, const std::string& suffix) {
        if (suffix.size() > word.size()) {
            return false;
        }
        return word.compare(word.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    // Replaces the suffix in 'word' with 'replacement' (if it ends with it)
    static void replace_suffix(std::string& word, const std::string& suffix, const std::string& replacement) {
        if (ends_with(word, suffix)) {
            word.replace(word.size() - suffix.size(), suffix.size(), replacement);
        }
    }

    // Main Porter stemming function/algorithm, if you will
    std::string porter_stem(const std::string& word_in) {
        if (word_in.length() <= 2) {
            return word_in; // Too short to stem
        }
        /*
        static const std::unordered_map<std::string, std::string> exceptions = {
            //{"programm", "program"},
            {"sky", "sky"},
            {"dying", "die"}
        };
        auto it = exceptions.find(word_in);
        if (it != exceptions.end()) return it->second;
        */

        std::string word = word_in;
        
        // Step 1a: Plural reduction
        if (ends_with(word, "sses")) {
            word.replace(word.size() - 4, 4, "ss");
        } else if (ends_with(word, "ies")) {
            word.replace(word.size() - 3, 3, "i");
        } else if (ends_with(word, "ss")) {
            // do nothing
        } else if (ends_with(word, "s")) {
            word.pop_back();
        }

        // Step 1b: Past tense and Gerunds
        if (ends_with(word, "eed")) {
            std::string stem = word.substr(0, word.size() - 3);
            if (measure(stem) > 0) {
                word.replace(word.size() - 3, 3, "ee");
            }
        } else if ((ends_with(word, "ed") && contains_vowel(word.substr(0, word.size() - 2)))) {
            word = word.substr(0, word.size() - 2);

            

            // After removing "ed", check if certain conditions apply
            if (ends_with(word, "at") || ends_with(word, "bl") || ends_with(word, "iz")) {
                word += "e";
            } else if (
                ends_with_double_consonant(word) &&
                word.back() != 'l' && word.back() != 's' && word.back() != 'z'
            ) {
                word.pop_back();
            } else if (
                measure(word) == 1 && 
                word.length() >= 3 &&
                is_consonant(word, word.length() - 1) &&
                !is_consonant(word, word.length() - 2) &&
                is_consonant(word, word.length() - 3) &&
                word.back() != 'w' && word.back() != 'x' && word.back() != 'y'
            ) {
                word += "e";
            }
        } else if ((ends_with(word, "ing") && contains_vowel(word.substr(0, word.size() - 3)))) {
            word = word.substr(0, word.size() - 3);

            // Made an attempt to apply the same cleanup logic somewhere above
            if (ends_with(word, "at") || ends_with(word, "bl") || ends_with(word, "iz")) {
                word += "e";
            } else if (
                ends_with_double_consonant(word) &&
                word.back() != 'l' &&  word.back() != 's' && word.back() != 'z'
            ) {
                word.pop_back();
            } else if (
                measure(word) == 1 && 
                word.length() >= 3 &&
                is_consonant(word, word.length() - 1) &&
                !is_consonant(word, word.length() - 2) &&
                is_consonant(word, word.length() - 3) &&
                word.back() != 'w' && word.back() != 'x' && word.back() != 'y'
            ) {
                word += "e";
            }
        }

       
        
        // Step 1c: 'y' to 'i' conversion
        if (ends_with(word, "y") && contains_vowel(word.substr(0, word.size() - 1))) {
            word.replace(word.size() - 1, 1, "i");
        }

        // Step 2: Replace commons suffixes if measure > 0
        struct Step2Suffix {
            const char* suffix;
            const char* replacement;
        } step2_suffixes[] = {
            {"ational", "ate"},
            {"tional", "tion"},
            {"enci", "ence"},
            {"anci", "ance"},
            {"izer", "ize"},
            {"abli", "able"},
            {"alli", "al"},
            {"entli", "ent"},
            {"eli", "e"},
            {"li", ""},
            {"ousli", "ous"},
            {"ization", "ize"},
            {"ation", "ate"},
            {"ator", "ate"},
            {"alism", "al"},
            {"iveness", "ive"},
            {"fulness", "ful"},
            {"ousness", "ous"},
            {"aliti", "al"},
            {"iviti", "ive"},
            {"biliti", "ble"},
            {"er", ""},
            {"mer", ""} // I guess I do not need this part...but oh well....
        };

        for (const auto& rule : step2_suffixes) {
            if (ends_with(word, rule.suffix)) {
                std::string stem = word.substr(0, word.size() - std::strlen(rule.suffix));
                if (measure(stem) > 0) {
                    word.replace(word.size() - std::strlen(rule.suffix), std::strlen(rule.suffix), rule.replacement);
                    break;
                }
            }
        }

        // Step 3: Replace other suffixes
        struct Step3Suffix {
            const char* suffix;
            const char* replacement;
        } step3_suffixes[] = {
            {"icate", "ic"},
            {"ative", ""},
            {"alize", "al"},
            {"iciti", "ic"},
            {"ical", "ic"},
            {"ful", ""},
            {"ness", ""},
        };

        for (const auto& rule : step3_suffixes) {
            if (ends_with(word, rule.suffix)) {
                std::string stem = word.substr(0, word.size() - std::strlen(rule.suffix));
                if (measure(stem) > 0) {
                    word.replace(word.size() - std::strlen(rule.suffix), std::strlen(rule.suffix), rule.replacement);
                    break;
                }
            }
        }

        // Step 4: Remove suffixes if measure > 1
        const char* step4_suffixes[] = {
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
            "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        };

        for (const char* suffix : step4_suffixes) {
            if (ends_with(word, suffix)) {
                std::string stem = word.substr(0, word.size() - std::strlen(suffix));
                if (measure(stem) > 1) {
                    if (std::string(suffix) == "ion") {

                        // Only remove "ion" if preceded by "s" or "t"
                        if (!stem.empty()) {
                            char ch = stem.back();
                            if (ch == 's' || ch == 't') {
                                word = stem;
                                break;
                            }
                        }
                    } else {
                        word = stem;
                        break;
                    }
                }
            }
        }

        // Step 5a: Final -e removal
        if (ends_with(word, "e")) {
            std::string stem = word.substr(0, word.size() - 1);
            if (measure(stem) > 1 || (measure(stem) == 1 && !ends_with(stem, "e"))) {
                word = stem;
            }
        }

        // Step 5b: Remove double 'l' if measure > 1
        if (measure(word) > 1 && ends_with(word, "ll")) {
            word.pop_back();
        }

        
        
        // This is just me trying to brute force certain parts....
        if (word == "plast") {
            word = "plaster";
        }
        
        if (word == "abl") {
            word = "abil";
        }
        
        
        return word;
    }
}