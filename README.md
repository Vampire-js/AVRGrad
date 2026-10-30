# VampireGrad embedded inference

The desktop project trains with `Tensor` and `Value`. Only the trained dense-layer parameters are exported; the autograd graph, gradients, STL containers, and `double` tensors are intentionally not copied to the Arduino Uno.

## Exporting a model

A deployable model uses the host-side structures in `tinygrad/include/model_export.h`:

- `DenseLayer`: rank-2 weights in `[output, input]` row-major order, rank-1 bias, and an activation.
- `InferenceModel`: an ordered vector of dense layers.
- `export_model_binary()`: writes a portable binary blob.
- `export_model_header()`: writes the same blob as an AVR `PROGMEM` C header.

Run `tinygrad/build.sh` to train the plant-watering example and regenerate `model_data.h`. An optional first argument selects another output path; a `.bin` suffix selects binary output.

The example is a one-neuron classifier with two binary inputs: `[soil_is_dry, weather_is_hot]`. It learns to recommend watering when either condition is true. On the Arduino, send two bits over UART: `00` means wet/cool and produces `SKIP`; `01`, `10`, and `11` produce `WATER`. A score of at least `0.5` is classified as `WATER`.

## Arduino API

`tinyinfer.h` is a heap-free C API. `tinyinfer_model_t` is a view over the serialized model rather than a copy, so generated weights remain in flash.

1. Call `tinyinfer_model_init()` once.
2. Allocate input/output arrays using `model.input_size` and `model.output_size`.
3. Allocate `tinyinfer_workspace_floats(&model)` floats of scratch memory.
4. Call `tinyinfer_predict()`.

The input, output, and workspace arrays must not overlap. The engine currently supports float32 dense layers with linear, ReLU, and sigmoid activations. The model's layer dimensions are validated when it is initialized.

## Model format (version 1)

All integers and floats are little-endian. Floats use IEEE-754 binary32.

- 16-byte model header: `VGRD`, version, dtype, layer count, flags, total byte size, reserved bytes.
- Per layer, an 8-byte header: input count, output count, activation, reserved bytes.
- Per layer payload: output-major float32 weights followed by float32 biases.

## Uno resource behavior

Weights generated in `model_data.h` use `PROGMEM` and consume flash, not SRAM. Runtime SRAM consists primarily of input, output, and two activation buffers. No dynamic allocation, exceptions, recursion, or autograd code is used on the microcontroller.

The included model remains a single dense neuron, keeping its flash and SRAM use small. Model arrays and activation workspaces add resource usage according to model dimensions.
