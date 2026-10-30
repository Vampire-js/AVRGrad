#define F_CPU 16000000UL

#include "model_data.h"
#include "tinyinfer.h"

#define UART_BAUD 57600UL
#include "uart.h"

#define IMAGE_PIXELS 784U
#define DIGIT_CLASSES 10U
#define FRAME_MAGIC_0 0x56U
#define FRAME_MAGIC_1 0x47U

static uint8_t image[IMAGE_PIXELS];
static int32_t scores[DIGIT_CLASSES];

static uint8_t receive_frame(void)
{
    uint16_t index;
    uint8_t checksum = 0U;

    for (;;) {
        if ((uint8_t)uart_receive() != FRAME_MAGIC_0) {
            continue;
        }
        if ((uint8_t)uart_receive() == FRAME_MAGIC_1) {
            break;
        }
    }

    for (index = 0U; index < IMAGE_PIXELS; ++index) {
        image[index] = (uint8_t)uart_receive();
        checksum = (uint8_t)(checksum + image[index]);
    }
    return checksum == (uint8_t)uart_receive();
}

int main(void)
{
    tinyinfer_model_t model;

    uart_init();

    if (tinyinfer_model_init(
            &model,
            vampiregrad_model,
            vampiregrad_model_size,
            TINYINFER_STORAGE_PROGMEM) != TINYINFER_OK) {
        uart_println("E:model");
        for (;;) {
        }
    }
    if (model.dtype != TINYINFER_DTYPE_INT8 ||
        model.input_size != IMAGE_PIXELS || model.output_size != DIGIT_CLASSES) {
        uart_println("E:shape");
        for (;;) {
        }
    }

    uart_println("READY");

    for (;;) {
        uint8_t prediction = 0U;
        uint8_t digit;

        if (!receive_frame()) {
            uart_println("E:checksum");
            continue;
        }

        if (tinyinfer_predict_u8(
                &model,
                image,
                IMAGE_PIXELS,
                scores,
                DIGIT_CLASSES) != TINYINFER_OK) {
            uart_println("E:inference");
            continue;
        }

        for (digit = 1U; digit < DIGIT_CLASSES; ++digit) {
            if (scores[digit] > scores[prediction]) {
                prediction = digit;
            }
        }
        uart_print("P:");
        uart_transmit((char)('0' + prediction));
        uart_println("");
    }
}