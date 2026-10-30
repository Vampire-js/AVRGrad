#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
class Tensor
{
public:
    Tensor(
        std::vector<std::size_t> shape,
        std::vector<double> data);
    [[nodiscard]] const std::vector<std::size_t> &shape() const noexcept;
    [[nodiscard]] const std::vector<double> &data() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] std::size_t dimension(const std::size_t axis) const;
    double &at(const std::vector<std::size_t> &idx);
    [[nodiscard]] double at(const std::vector<std::size_t> &idx) const;
    [[nodiscard]] double sum() const noexcept;

    void reshape(std::vector<std::size_t> shape);
    void transpose();
    friend Tensor operator*(const Tensor &a, const Tensor &b);
    friend Tensor operator+(const Tensor &a, const Tensor &b);
    friend Tensor operator-(const Tensor &a, const Tensor &b);

    Tensor &operator*=(const Tensor &a);
    Tensor &operator+=(const Tensor &a);
    Tensor &operator-=(const Tensor &a);
    Tensor &operator=(double b);
private:
    std::vector<std::size_t> shape_;
    std::vector<double> data_;
    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t> &idx) const;
};

