#pragma once

#include "tensor.h"

#include <cstdint>
#include <string>
#include <vector>

enum class Activation : std::uint8_t
{
    linear = 0,
    relu = 1,
    sigmoid = 2
};

/* Weights use row-major [output, input] layout; bias has shape [output]. */
struct DenseLayer
{
    Tensor weights;
    Tensor bias;
    Activation activation = Activation::linear;
};

struct InferenceModel
{
    std::vector<DenseLayer> layers;
};

/* Export the portable little-endian float32 format consumed by tinyinfer. */
void export_model_binary(const InferenceModel &model, const std::string &path);

/* Export the same bytes as an AVR PROGMEM-ready C header. */
void export_model_header(
    const InferenceModel &model,
    const std::string &path,
    const std::string &symbol = "vampiregrad_model");
