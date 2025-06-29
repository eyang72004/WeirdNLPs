cmake_minimum_required(VERSION 3.15)

project(WeirdNLPpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Local LibTorch path
set(Torch_DIR "/Users/ethanyang/Downloads/WeirdNLPs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# I think this is needed for PyBind11 but...
find_package(pybind11 REQUIRED)

# == Include Directories ==
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# == Core C++ Library ==
file(GLOB SOURCES "src/*.cpp")
add_library(weirdNLPpp ${SOURCES})
target_link_libraries(weirdNLPpp "${TORCH_LIBRARIES}")


# == Test Executables ==
add_executable(test_tokenization tests/test_tokenization.cpp)
target_link_libraries(test_tokenization weirdNLPpp)

add_executable(test_stemming tests/test_stemming.cpp)
target_link_libraries(test_stemming weirdNLPpp)

add_executable(test_lemmatization tests/test_lemmatization.cpp)
target_link_libraries(test_lemmatization weirdNLPpp)

add_executable(test_corpus tests/test_corpus.cpp)
target_link_libraries(test_corpus weirdNLPpp)

add_executable(test_vectorization tests/test_vectorization.cpp)
target_link_libraries(test_vectorization weirdNLPpp)

add_executable(test_embeddings tests/test_embeddings.cpp)
target_link_libraries(test_embeddings weirdNLPpp)

add_executable(test_syntax tests/test_syntax.cpp)
target_link_libraries(test_syntax weirdNLPpp)

add_executable(test_ner tests/test_ner.cpp)
target_link_libraries(test_ner weirdNLPpp)

add_executable(test_sentiment tests/test_sentiment.cpp)
target_link_libraries(test_sentiment weirdNLPpp)

add_executable(test_ml_models tests/test_ml_models.cpp)
target_link_libraries(test_ml_models weirdNLPpp)

add_executable(test_deep_models tests/test_deep_models.cpp)
target_link_libraries(test_deep_models weirdNLPpp "${TORCH_LIBRARIES}")
set_property(TARGET test_deep_models PROPERTY CXX_STANDARD 17)

add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils weirdNLPpp)

add_executable(test_markov_chain tests/test_markov_chain.cpp)
target_link_libraries(test_markov_chain weirdNLPpp)

add_executable(example_pipeline examples/example_pipeline.cpp)
target_link_libraries(example_pipeline weirdNLPpp)



# === Python and PyBind11 Buildings Hopefully ===
add_library(weirdnlp MODULE bindings/bindings.cpp)
target_link_libraries(weirdnlp PRIVATE pybind11::module weirdNLPpp "${TORCH_LIBRARIES}")



set_target_properties(weirdnlp PROPERTIES
    PREFIX ""                           
    OUTPUT_NAME "weirdnlp"              
    SUFFIX ".so"                        # Ok....I am actually not sure why this will not make me the .so file.....
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/weirdnlp"  # So it ends up in a place that never ends up at that place...
)



#add_subdirectory(bindings)