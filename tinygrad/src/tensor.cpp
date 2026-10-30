#include "tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

Tensor::Tensor(
    std::vector<std::size_t> shape,
    std::vector<double> data)
    : shape_(std::move(shape)), data_(std::move(data))
{
    std::size_t expected_elements;

    bool has_zero_dims = false;

    for (const std::size_t dimension : shape_)
    {
        if (dimension == 0)
        {
            has_zero_dims = true;
            expected_elements = 0;
            break;
        }
    }

    if (!has_zero_dims)
    {
        expected_elements = 1;
        for (const std::size_t dimension : shape_)
        {
            if (expected_elements > std::numeric_limits<std::size_t>::max() / dimension)
            {
                throw std::overflow_error("tensor element count overflows size_t");
            }

            expected_elements *= dimension;
        }
    }

    if (expected_elements != data_.size())
    {
        throw std::invalid_argument("tensor shape does not match its data");
    }
}

const std::vector<std::size_t> &Tensor::shape() const noexcept
{
    return shape_;
}

const std::vector<double> &Tensor::data() const noexcept
{
    return data_;
}

std::size_t Tensor::rank() const noexcept
{
    return shape_.size();
}

std::size_t Tensor::numel() const noexcept
{
    return data_.size();
}

std::size_t Tensor::dimension(const std::size_t axis) const
{
    if (axis >= rank())
    {
        throw std::out_of_range("tensor axis is outside its rank");
    }
    return shape_[axis];
}

double &Tensor::at(const std::vector<std::size_t> &idx)
{
    return data_[flat_index(idx)];
}

double Tensor::at(const std::vector<std::size_t> &idx) const
{
    return data_[flat_index(idx)];
}

double Tensor::sum() const noexcept
{
    double result = 0.0;
    for (const double value : data_)
    {
        result += value;
    }

    return result;
}

void Tensor::reshape(std::vector<std::size_t> shape)
{
    std::size_t expected = 1;
    for (int i : shape)
        expected *= i;
    if (expected == numel() && shape.size() == rank())
        shape_ = std::move(shape);
    else
        throw std::invalid_argument("Incorrect shape");
}

void Tensor::transpose()
{
    std::reverse(shape_.begin(), shape_.end());
}

Tensor &Tensor::operator*=(const Tensor &a)
{
    *this = *this * a;
    return *this;
}

Tensor &Tensor::operator+=(const Tensor &a)
{
    *this = *this + a;
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &a)
{
    *this = *this - a;
    return *this;
}

Tensor &Tensor::operator=(double b)
{
    std::fill(data_.begin(), data_.end(), b);
    return *this;
}

std::size_t Tensor::flat_index(const std::vector<std::size_t> &idx) const
{
    if (idx.size() != rank())
    {
        throw std::invalid_argument("number of indices must match tensor rank");
    }

    for (std::size_t axis = 0; axis < rank(); ++axis)
    {
        if (idx[axis] >= shape_[axis])
        {
            throw std::out_of_range("tensor index is outside its dimension");
        }
    }

    std::size_t flat_index = 0;
    std::size_t stride = 1;

    for (std::size_t axis = rank(); axis > 0; --axis)
    {
        const std::size_t current_axis = axis - 1;
        flat_index += idx[current_axis] * stride;
        stride *= shape_[current_axis];
    }

    return flat_index;
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
        throw std::invalid_argument("Operands cannot be multiplied element wise");
    std::vector<double> data(a.numel());
    for (std::size_t i = 0; i < a.numel(); i++)
        data[i] = a.data_[i] * b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}

Tensor operator+(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
        throw std::invalid_argument("Operands cannot be added element wise");
    std::vector<double> data(a.numel());
    for (std::size_t i = 0; i < a.numel(); i++)
        data[i] = a.data_[i] + b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}

Tensor operator-(const Tensor &a, const Tensor &b)
{
    if (a.shape() != b.shape())
        throw std::invalid_argument("Operands cannot be subtracted element wise");
    std::vector<double> data(a.numel());
    for (std::size_t i = 0; i < a.numel(); i++)
        data[i] = a.data_[i] - b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}

