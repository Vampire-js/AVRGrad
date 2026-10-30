#include <iostream>
#include <string>

#include "model_export.h"
#include "tensor_autograd.h"

int main(int argc, char **argv)
{
    // y = 2x + 1
    const double x_data[] = {1.0, 2.0, 3.0, 4.0};
    const double y_data[] = {3.0, 5.0, 7.0, 9.0};
    constexpr std::size_t sample_count = 4;
    constexpr std::size_t epochs = 20000;
    constexpr double learning_rate = 0.01;

    Value weight(Tensor({}, {0.0}));
    Value bias(Tensor({}, {0.0}));
    double final_loss = 0.0;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        Value squared_error_sum(Tensor({}, {0.0}));

        for (std::size_t i = 0; i < sample_count; ++i)
        {
            const Value x(Tensor({}, {x_data[i]}));
            const Value target(Tensor({}, {y_data[i]}));
            const Value prediction = x * weight + bias;
            const Value error = prediction - target;
            squared_error_sum = squared_error_sum + error * error;
        }

        const Value mean_scale(Tensor({}, {1.0 / sample_count}));
        Value mse = squared_error_sum * mean_scale;
        mse.backward();
        final_loss = mse.data().at({});

        weight.step(learning_rate);
        bias.step(learning_rate);
        weight.zero_grad();
        bias.zero_grad();
    }

    std::cout << "MSE: " << final_loss << '\n';
    std::cout << "weight: " << weight.data().at({}) << '\n';
    std::cout << "bias: " << bias.data().at({}) << '\n';

    if (argc > 1)
    {
        const InferenceModel model{{DenseLayer{
            Tensor({1, 1}, {weight.data().at({})}),
            Tensor({1}, {bias.data().at({})}),
            Activation::linear}}};
        const std::string path = argv[1];
        if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin")
        {
            export_model_binary(model, path);
        }
        else
        {
            export_model_header(model, path);
        }
        std::cout << "exported model: " << path << '\n';
    }
    return 0;
}