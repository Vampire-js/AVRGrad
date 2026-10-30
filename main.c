#define F_CPU 16000000UL

#include <stdlib.h>

#include "model_data.h"
#include "tinyinfer.h"

#define UART_BAUD 9600UL
#include "uart.h"

int main(void)
{
    tinyinfer_model_t model;
    float workspace[2];
    float input[1];
    float output[1];
    char formatted[16];

    uart_init();

    if (tinyinfer_model_init(
            &model,
            vampiregrad_model,
            vampiregrad_model_size,
            TINYINFER_STORAGE_PROGMEM) != TINYINFER_OK) {
        uart_println("Invalid model");
        for (;;) {
        }
    }

    uart_println("VampireGrad model ready");
    uart_println("Send one digit (0-9):");

    for (;;) {
        const char c = uart_receive();
        if (c < '0' || c > '9') {
            continue;
        }

        input[0] = (float)(c - '0');
        if (tinyinfer_predict(
                &model,
                input,
                1U,
                output,
                1U,
                workspace,
                sizeof(workspace) / sizeof(workspace[0])) != TINYINFER_OK) {
            uart_println("Inference failed");
            continue;
        }

        dtostrf(output[0], 0, 3, formatted);
        uart_print("prediction: ");
        uart_println(formatted);
    }
}