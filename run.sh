
#!/bin/bash
set -e

MCU="atmega328p"
PORT="/dev/ttyACM0"
BAUD="115200"
ELF="firmware.elf"
HEX="firmware.hex"

# sudo pkill picocom

echo "==> Compiling..."
avr-gcc \
    -mmcu=$MCU \
    -DF_CPU=16000000UL \
    -Os \
    -Wall -Wextra \
    main.c tinyinfer.c \
    -lm \
    -o "$ELF"

echo "==> Generating HEX..."
sudo avr-objcopy -O ihex -R .eeprom "$ELF" "$HEX"

echo "==> Memory usage..."
avr-size -C --mcu=$MCU "$ELF"

echo "==> Uploading to Arduino..."
sudo avrdude \
    -p $MCU \
    -c arduino \
    -P "$PORT" \
    -b $BAUD \
    -D \
    -U flash:w:"$HEX":i

echo "==> Upload complete!"
