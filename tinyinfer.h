#ifndef TINYINFER_H
#define TINYINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINYINFER_FORMAT_VERSION 1U
#define TINYINFER_HEADER_SIZE 16U
#define TINYINFER_LAYER_HEADER_SIZE 8U

typedef enum {
    TINYINFER_STORAGE_RAM = 0,
    TINYINFER_STORAGE_PROGMEM = 1
} tinyinfer_storage_t;

typedef enum {
    TINYINFER_ACTIVATION_LINEAR = 0,
    TINYINFER_ACTIVATION_RELU = 1,
    TINYINFER_ACTIVATION_SIGMOID = 2
} tinyinfer_activation_t;

typedef enum {
    TINYINFER_OK = 0,
    TINYINFER_INVALID_ARGUMENT,
    TINYINFER_INVALID_MODEL,
    TINYINFER_UNSUPPORTED_VERSION,
    TINYINFER_UNSUPPORTED_TYPE,
    TINYINFER_BUFFER_TOO_SMALL
} tinyinfer_status_t;

typedef struct {
    const uint8_t *data;
    size_t size;
    tinyinfer_storage_t storage;
    uint8_t layer_count;
    uint16_t input_size;
    uint16_t output_size;
    uint16_t max_width;
} tinyinfer_model_t;

/* Validate a serialized model and initialize a lightweight read-only view. */
tinyinfer_status_t tinyinfer_model_init(
    tinyinfer_model_t *model,
    const void *data,
    size_t size,
    tinyinfer_storage_t storage);

/* Number of floats required by tinyinfer_predict(). */
size_t tinyinfer_workspace_floats(const tinyinfer_model_t *model);

/* Run all dense layers. input, output, and workspace must not overlap. */
tinyinfer_status_t tinyinfer_predict(
    const tinyinfer_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count,
    float *workspace,
    size_t workspace_count);

#ifdef __cplusplus
}
#endif

#endif
