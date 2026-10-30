#pragma once

#include "tensor.h"

#include <memory>
#include <unordered_set>
#include <vector>

struct Node;

class Value
{
public:
    explicit Value(Tensor data);

    [[nodiscard]] Tensor data() const noexcept;
    [[nodiscard]] Tensor grad() const noexcept;

    friend Value operator+(const Value &left, const Value &right);
    friend Value operator-(const Value &left, const Value &right);
    friend Value operator*(const Value &left, const Value &right);

    void backward();
    void step(double learning_rate);
    void zero_grad();

private:
    std::shared_ptr<Node> node_;

    explicit Value(std::shared_ptr<Node> node);

    static void build_topology(
        Node *node,
        std::unordered_set<Node *> &visited,
        std::vector<Node *> &topology);
};