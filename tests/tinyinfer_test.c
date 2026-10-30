#include "model_data.h"
#include "tinyinfer.h"

#include <assert.h>
#include <string.h>

int main(void)
{
    tinyinfer_model_t model;
    uint8_t image[784];
    int32_t scores[10];
    size_t digit;
    size_t prediction = 0U;

    memset(image, 0, sizeof(image));

    assert(tinyinfer_model_init(
               &model,
               vampiregrad_model,
               vampiregrad_model_size,
               TINYINFER_STORAGE_RAM) == TINYINFER_OK);
    assert(model.dtype == TINYINFER_DTYPE_INT8);
    assert(model.input_size == 784U);
    assert(model.output_size == 10U);
    assert(tinyinfer_predict_u8(
               &model,
               image,
               784U,
               scores,
               10U) == TINYINFER_OK);
    for (digit = 1U; digit < 10U; ++digit) {
        if (scores[digit] > scores[prediction]) {
            prediction = digit;
        }
    }
    assert(prediction == 5U);
    return 0;
}
