#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "tensor.h"
#include "matmul.h"

Tensor matmul(const Tensor &a, Tensor b)
{
    if (a.rank() > 2 || b.rank() > 2)
        throw std::invalid_argument("Input operands aren't matrices");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("Input tensor shapes cannot be multiplied");
    std::vector<double> data(a.shape()[0] * b.shape()[1], 0);
    std::size_t l = a.shape()[0];
    std::size_t m = a.shape()[1];
    std::size_t n = b.shape()[1];

    auto flatten = [n](std::size_t i, std::size_t j){
        return (i * n + j);
    };

    for (std::size_t i = 0; i < l; i++){
        for (std::size_t j = 0; j < n; j++){
            for (std::size_t k = 0; k < m; k++){
                data[flatten(i, j)] += a.at({i, k}) * b.at({k, j});
            }
        }
    }
    return Tensor({a.shape()[0], b.shape()[1]}, {data});
}
