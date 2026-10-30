#include "model_export.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace
{
constexpr std::uint8_t format_version = 1;
constexpr std::uint8_t dtype_float32 = 1;

void append_u16(std::vector<std::uint8_t> &bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
}

void append_u32(std::vector<std::uint8_t> &bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
    {
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_f32(std::vector<std::uint8_t> &bytes, double value)
{
    const float narrowed = static_cast<float>(value);
    std::uint32_t bits;
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::memcpy(&bits, &narrowed, sizeof(bits));
    append_u32(bytes, bits);
}

std::vector<std::uint8_t> serialize(const InferenceModel &model)
{
    if (model.layers.empty() || model.layers.size() > std::numeric_limits<std::uint8_t>::max())
    {
        throw std::invalid_argument("model must contain between 1 and 255 layers");
    }

    std::vector<std::uint8_t> bytes = {
        'V', 'G', 'R', 'D', format_version, dtype_float32,
        static_cast<std::uint8_t>(model.layers.size()), 0,
        0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t previous_outputs = 0;

    for (std::size_t index = 0; index < model.layers.size(); ++index)
    {
        const DenseLayer &layer = model.layers[index];
        if (layer.weights.rank() != 2 || layer.bias.rank() != 1)
        {
            throw std::invalid_argument("dense weights must be rank 2 and bias rank 1");
        }

        const std::size_t outputs = layer.weights.dimension(0);
        const std::size_t inputs = layer.weights.dimension(1);
        if (inputs == 0 || outputs == 0 || layer.bias.dimension(0) != outputs ||
            inputs > std::numeric_limits<std::uint16_t>::max() ||
            outputs > std::numeric_limits<std::uint16_t>::max())
        {
            throw std::invalid_argument("invalid dense layer dimensions");
        }
        if (index != 0 && inputs != previous_outputs)
        {
            throw std::invalid_argument("adjacent dense layer dimensions do not match");
        }
        if (static_cast<std::uint8_t>(layer.activation) > static_cast<std::uint8_t>(Activation::sigmoid))
        {
            throw std::invalid_argument("unsupported activation");
        }

        append_u16(bytes, static_cast<std::uint16_t>(inputs));
        append_u16(bytes, static_cast<std::uint16_t>(outputs));
        bytes.push_back(static_cast<std::uint8_t>(layer.activation));
        bytes.push_back(0);
        append_u16(bytes, 0);
        for (double weight : layer.weights.data())
        {
            append_f32(bytes, weight);
        }
        for (double bias : layer.bias.data())
        {
            append_f32(bytes, bias);
        }
        previous_outputs = outputs;
    }

    if (bytes.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("serialized model is too large");
    }
    const std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
    for (unsigned byte = 0; byte < 4; ++byte)
    {
        bytes[8 + byte] = static_cast<std::uint8_t>(size >> (byte * 8));
    }
    return bytes;
}

std::string validate_symbol(const std::string &symbol)
{
    if (symbol.empty() || (!std::isalpha(static_cast<unsigned char>(symbol.front())) && symbol.front() != '_'))
    {
        throw std::invalid_argument("model symbol is not a valid C identifier");
    }
    for (char c : symbol)
    {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
        {
            throw std::invalid_argument("model symbol is not a valid C identifier");
        }
    }
    return symbol;
}
} // namespace

void export_model_binary(const InferenceModel &model, const std::string &path)
{
    const std::vector<std::uint8_t> bytes = serialize(model);
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        throw std::runtime_error("could not open model output: " + path);
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::runtime_error("could not write model output: " + path);
    }
}

void export_model_header(
    const InferenceModel &model,
    const std::string &path,
    const std::string &symbol)
{
    const std::vector<std::uint8_t> bytes = serialize(model);
    const std::string checked_symbol = validate_symbol(symbol);
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("could not open model header: " + path);
    }

    output << "#ifndef VAMPIREGRAD_GENERATED_MODEL_H\n"
              "#define VAMPIREGRAD_GENERATED_MODEL_H\n\n"
              "#include <stddef.h>\n"
              "#include <stdint.h>\n\n"
              "#ifdef __AVR__\n"
              "#include <avr/pgmspace.h>\n"
              "#define VAMPIREGRAD_MODEL_DATA PROGMEM\n"
              "#else\n"
              "#define VAMPIREGRAD_MODEL_DATA\n"
              "#endif\n\n";
    output << "static const uint8_t " << checked_symbol << "[] VAMPIREGRAD_MODEL_DATA = {\n    ";
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        output << "0x" << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<unsigned>(bytes[i]);
        if (i + 1 != bytes.size())
        {
            output << ", ";
        }
        if ((i + 1) % 12 == 0 && i + 1 != bytes.size())
        {
            output << "\n    ";
        }
    }
    output << std::dec << "\n};\n\n"
           << "#define " << checked_symbol << "_size ((size_t)" << bytes.size() << "U)\n\n"
              "#undef VAMPIREGRAD_MODEL_DATA\n"
              "#endif\n";
    if (!output)
    {
        throw std::runtime_error("could not write model header: " + path);
    }
}
