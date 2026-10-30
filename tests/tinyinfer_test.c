#include "model_data.h"
#include "tinyinfer.h"

#include <assert.h>
#include <math.h>

int main(void)
{
    tinyinfer_model_t model;
    float input[] = {4.0f};
    float output[1];
    float workspace[2];

    assert(tinyinfer_model_init(
               &model,
               vampiregrad_model,
               vampiregrad_model_size,
               TINYINFER_STORAGE_RAM) == TINYINFER_OK);
    assert(model.input_size == 1U);
    assert(model.output_size == 1U);
    assert(tinyinfer_workspace_floats(&model) == 1U);
    assert(tinyinfer_predict(
               &model,
               input,
               1U,
               output,
               1U,
               workspace,
               2U) == TINYINFER_OK);
    assert(fabsf(output[0] - 9.0f) < 0.001f);
    return 0;
}
