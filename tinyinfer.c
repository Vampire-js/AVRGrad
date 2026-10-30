#include "tinyinfer.h"

#include <math.h>

#ifdef __AVR__
#include <avr/pgmspace.h>
#endif

#define TINYINFER_MAGIC_0 'V'
#define TINYINFER_MAGIC_1 'G'
#define TINYINFER_MAGIC_2 'R'
#define TINYINFER_MAGIC_3 'D'

static uint8_t read_u8(const tinyinfer_model_t *model, size_t offset)
{
#ifdef __AVR__
    if (model->storage == TINYINFER_STORAGE_PROGMEM) {
        return pgm_read_byte(model->data + offset);
    }
#else
    (void)model->storage;
#endif
    return model->data[offset];
}

static uint16_t read_u16(const tinyinfer_model_t *model, size_t offset)
{
    return (uint16_t)read_u8(model, offset) |
           ((uint16_t)read_u8(model, offset + 1U) << 8U);
}

static uint32_t read_u32(const tinyinfer_model_t *model, size_t offset)
{
    return (uint32_t)read_u8(model, offset) |
           ((uint32_t)read_u8(model, offset + 1U) << 8U) |
           ((uint32_t)read_u8(model, offset + 2U) << 16U) |
           ((uint32_t)read_u8(model, offset + 3U) << 24U);
}

static int32_t read_i32(const tinyinfer_model_t *model, size_t offset)
{
    return (int32_t)read_u32(model, offset);
}

static float read_f32(const tinyinfer_model_t *model, size_t offset)
{
    union {
        uint32_t bits;
        float value;
    } decoded;

    decoded.bits = read_u32(model, offset);
    return decoded.value;
}

static float activate(float value, uint8_t activation)
{
    switch (activation) {
    case TINYINFER_ACTIVATION_RELU:
        return value > 0.0f ? value : 0.0f;
    case TINYINFER_ACTIVATION_SIGMOID:
        return 1.0f / (1.0f + expf(-value));
    default:
        return value;
    }
}

tinyinfer_status_t tinyinfer_model_init(
    tinyinfer_model_t *model,
    const void *data,
    size_t size,
    tinyinfer_storage_t storage)
{
    size_t offset = TINYINFER_HEADER_SIZE;
    uint16_t previous_outputs = 0U;
    uint16_t max_width = 0U;
    uint8_t layer;

    if (model == NULL || data == NULL || size < TINYINFER_HEADER_SIZE) {
        return TINYINFER_INVALID_ARGUMENT;
    }
    if (storage != TINYINFER_STORAGE_RAM && storage != TINYINFER_STORAGE_PROGMEM) {
        return TINYINFER_INVALID_ARGUMENT;
    }

    model->data = (const uint8_t *)data;
    model->size = size;
    model->storage = storage;
    model->dtype = (tinyinfer_dtype_t)0;
    model->layer_count = 0U;
    model->input_size = 0U;
    model->output_size = 0U;
    model->max_width = 0U;

    if (read_u8(model, 0U) != TINYINFER_MAGIC_0 ||
        read_u8(model, 1U) != TINYINFER_MAGIC_1 ||
        read_u8(model, 2U) != TINYINFER_MAGIC_2 ||
        read_u8(model, 3U) != TINYINFER_MAGIC_3) {
        return TINYINFER_INVALID_MODEL;
    }
    if (read_u8(model, 4U) != TINYINFER_FORMAT_VERSION) {
        return TINYINFER_UNSUPPORTED_VERSION;
    }
    model->dtype = (tinyinfer_dtype_t)read_u8(model, 5U);
    if (model->dtype != TINYINFER_DTYPE_FLOAT32 && model->dtype != TINYINFER_DTYPE_INT8) {
        return TINYINFER_UNSUPPORTED_TYPE;
    }

    model->layer_count = read_u8(model, 6U);
    if (model->layer_count == 0U || read_u8(model, 7U) != 0U ||
        read_u32(model, 8U) != (uint32_t)size) {
        return TINYINFER_INVALID_MODEL;
    }

    for (layer = 0U; layer < model->layer_count; ++layer) {
        uint16_t inputs;
        uint16_t outputs;
        uint8_t activation;
        size_t payload_size;

        if (offset > size || size - offset < TINYINFER_LAYER_HEADER_SIZE) {
            return TINYINFER_INVALID_MODEL;
        }

        inputs = read_u16(model, offset);
        outputs = read_u16(model, offset + 2U);
        activation = read_u8(model, offset + 4U);
        if (inputs == 0U || outputs == 0U || activation > TINYINFER_ACTIVATION_SIGMOID ||
            read_u8(model, offset + 5U) != 0U || read_u16(model, offset + 6U) != 0U) {
            return TINYINFER_INVALID_MODEL;
        }
        if (model->dtype == TINYINFER_DTYPE_INT8 &&
            (model->layer_count != 1U || activation != TINYINFER_ACTIVATION_LINEAR)) {
            return TINYINFER_INVALID_MODEL;
        }
        if (layer != 0U && inputs != previous_outputs) {
            return TINYINFER_INVALID_MODEL;
        }
        if ((size_t)inputs > SIZE_MAX / (size_t)outputs) {
            return TINYINFER_INVALID_MODEL;
        }
        payload_size = (size_t)inputs * (size_t)outputs;
        if (payload_size > SIZE_MAX - outputs) {
            return TINYINFER_INVALID_MODEL;
        }
        if (model->dtype == TINYINFER_DTYPE_FLOAT32) {
            payload_size += outputs;
            if (payload_size > SIZE_MAX / sizeof(float)) {
                return TINYINFER_INVALID_MODEL;
            }
            payload_size *= sizeof(float);
        } else {
            if (payload_size > SIZE_MAX - sizeof(float) ||
                (size_t)outputs > (SIZE_MAX - payload_size - sizeof(float)) / sizeof(int32_t)) {
                return TINYINFER_INVALID_MODEL;
            }
            payload_size += sizeof(float) + (size_t)outputs * sizeof(int32_t);
        }
        offset += TINYINFER_LAYER_HEADER_SIZE;
        if (offset > size || payload_size > size - offset) {
            return TINYINFER_INVALID_MODEL;
        }
        offset += payload_size;

        if (layer == 0U) {
            model->input_size = inputs;
        }
        previous_outputs = outputs;
        if (outputs > max_width) {
            max_width = outputs;
        }
    }

    if (offset != size) {
        return TINYINFER_INVALID_MODEL;
    }

    model->output_size = previous_outputs;
    model->max_width = max_width;
    return TINYINFER_OK;
}

size_t tinyinfer_workspace_floats(const tinyinfer_model_t *model)
{
    if (model == NULL || model->layer_count == 0U) {
        return 0U;
    }
    return model->layer_count == 1U ? model->max_width : (size_t)model->max_width * 2U;
}

tinyinfer_status_t tinyinfer_predict(
    const tinyinfer_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    float *workspace,
    size_t workspace_count)
{
    const float *current = input;
    size_t offset = TINYINFER_HEADER_SIZE;
    uint8_t layer;

    if (model == NULL || input == NULL || output == NULL || workspace == NULL ||
        model->layer_count == 0U) {
        return TINYINFER_INVALID_ARGUMENT;
    }
    if (model->dtype != TINYINFER_DTYPE_FLOAT32) {
        return TINYINFER_UNSUPPORTED_TYPE;
    }
    if (input_count != model->input_size || output_count < model->output_size ||
        workspace_count < tinyinfer_workspace_floats(model)) {
        return TINYINFER_BUFFER_TOO_SMALL;
    }

    for (layer = 0U; layer < model->layer_count; ++layer) {
        const uint16_t inputs = read_u16(model, offset);
        const uint16_t outputs = read_u16(model, offset + 2U);
        const uint8_t activation = read_u8(model, offset + 4U);
        const size_t weights_offset = offset + TINYINFER_LAYER_HEADER_SIZE;
        const size_t biases_offset = weights_offset + (size_t)inputs * outputs * sizeof(float);
        float *next = workspace + ((layer & 1U) != 0U ? model->max_width : 0U);
        uint16_t out_index;

        for (out_index = 0U; out_index < outputs; ++out_index) {
            float sum = read_f32(model, biases_offset + (size_t)out_index * sizeof(float));
            uint16_t in_index;

            for (in_index = 0U; in_index < inputs; ++in_index) {
                const size_t weight_index = (size_t)out_index * inputs + in_index;
                sum += current[in_index] * read_f32(model, weights_offset + weight_index * sizeof(float));
            }
            next[out_index] = activate(sum, activation);
        }

        current = next;
        offset = biases_offset + (size_t)outputs * sizeof(float);
    }

    for (size_t i = 0U; i < model->output_size; ++i) {
        output[i] = current[i];
    }
    return TINYINFER_OK;
}

tinyinfer_status_t tinyinfer_predict_u8(
    const tinyinfer_model_t *model,
    const uint8_t *input,
    size_t input_count,
    int32_t *scores,
    size_t score_count)
{
    const size_t layer_offset = TINYINFER_HEADER_SIZE;
    uint16_t inputs;
    uint16_t outputs;
    size_t weights_offset;
    size_t biases_offset;
    uint16_t out_index;

    if (model == NULL || input == NULL || scores == NULL) {
        return TINYINFER_INVALID_ARGUMENT;
    }
    if (model->dtype != TINYINFER_DTYPE_INT8) {
        return TINYINFER_UNSUPPORTED_TYPE;
    }
    if (input_count != model->input_size || score_count < model->output_size) {
        return TINYINFER_BUFFER_TOO_SMALL;
    }

    inputs = read_u16(model, layer_offset);
    outputs = read_u16(model, layer_offset + 2U);
    /* The first four payload bytes hold the host-side dequantization scale. */
    weights_offset = layer_offset + TINYINFER_LAYER_HEADER_SIZE + sizeof(float);
    biases_offset = weights_offset + (size_t)inputs * outputs;

    for (out_index = 0U; out_index < outputs; ++out_index) {
        int32_t accumulator = read_i32(model, biases_offset + (size_t)out_index * sizeof(int32_t));
        uint16_t in_index;

        for (in_index = 0U; in_index < inputs; ++in_index) {
            const size_t weight_index = (size_t)out_index * inputs + in_index;
            const int8_t weight = (int8_t)read_u8(model, weights_offset + weight_index);
            accumulator += (int32_t)input[in_index] * weight;
        }
        scores[out_index] = accumulator;
    }
    return TINYINFER_OK;
}
