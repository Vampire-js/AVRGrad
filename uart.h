#ifndef UART_H
#define UART_H

#include <avr/io.h>
#include <stddef.h>
#include <stdint.h>

#ifndef F_CPU
#error "F_CPU must be defined before including uart.h"
#endif

#ifndef UART_BAUD
#define UART_BAUD 9600UL
#endif

#define UART_UBRR_VALUE ((F_CPU + (UART_BAUD * 8UL)) / (UART_BAUD * 16UL) - 1UL)

#if UART_UBRR_VALUE > 4095UL
#error "UART_BAUD is too low for this CPU clock"
#endif

static inline void uart_init(void)
{
    const uint16_t ubrr = (uint16_t)UART_UBRR_VALUE;

    UBRR0H = (uint8_t)(ubrr >> 8);
    UBRR0L = (uint8_t)ubrr;

    /* Normal speed, no multiprocessor mode, and clear pending TX-complete. */
    UCSR0A = (1U << TXC0);
    UCSR0B = (1U << RXEN0) | (1U << TXEN0);
    UCSR0C = (1U << UCSZ01) | (1U << UCSZ00); /* 8 data bits, no parity, 1 stop bit. */
}

static inline uint8_t uart_rx_ready(void)
{
    return (UCSR0A & (1U << RXC0)) != 0U;
}

static inline uint8_t uart_tx_ready(void)
{
    return (UCSR0A & (1U << UDRE0)) != 0U;
}

static inline void uart_transmit(char c)
{
    while (!uart_tx_ready()) {
    }

    UDR0 = (uint8_t)c;
}

static inline char uart_receive(void)
{
    while (!uart_rx_ready()) {
    }

    return (char)UDR0;
}

/* Returns 1 when a byte was read, or 0 when no byte is currently available. */
static inline uint8_t uart_try_receive(char *c)
{
    if (c == NULL || !uart_rx_ready()) {
        return 0U;
    }

    *c = (char)UDR0;
    return 1U;
}

static inline void uart_write(const void *data, size_t length)
{
    const uint8_t *bytes = (const uint8_t *)data;

    if (bytes == NULL) {
        return;
    }

    while (length-- > 0U) {
        uart_transmit((char)*bytes++);
    }
}

static inline void uart_print(const char *str)
{
    if (str == NULL) {
        return;
    }

    while (*str != '\0') {
        uart_transmit(*str++);
    }
}

static inline void uart_println(const char *str)
{
    uart_print(str);
    uart_transmit('\r');
    uart_transmit('\n');
}

/* Wait until the final byte has completely left the shift register. */
static inline void uart_flush(void)
{
    while ((UCSR0A & (1U << TXC0)) == 0U) {
    }

    UCSR0A = (1U << TXC0);
}

#endif
