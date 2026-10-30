#include "tensor_autograd.h"

#include <unordered_set>
#include <algorithm>
#include <utility>
#include <vector>

enum class Operation {
    leaf,
    add,
    subtract,
    multiply
};


struct Node {
    Tensor data; // result of forward computation
    Tensor grad; // relative change in loss wrt current node
    Operation opereation = Operation::leaf;
    std::vector<std::shared_ptr<Node>> parents; // inputs that produced this node

    explicit Node(Tensor value)
        : data(std::move(value)), grad(data.shape(), std::vector<double>(data.numel(), 0.0)) {}
};

Value::Value(Tensor data) : node_(std::make_shared<Node>(std::move(data))) {}

Value::Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}

Tensor Value::data() const noexcept
{
    return node_->data;
}

Tensor Value::grad() const noexcept
{
    return node_->grad;
}

void Value::backward()
{
    std::unordered_set<Node *> visited;
    std::vector<Node *> topology;
    build_topology(node_.get(), visited, topology);

    node_->grad = 1.0;
    std::reverse(topology.begin(), topology.end());
    for (auto curr = topology.begin(); curr != topology.end(); curr++)
    {
        Node *output = *curr;
        switch (output->opereation)
        {
        case Operation::leaf:
            break;
        case Operation::add:
            output->parents[0]->grad += output->grad;
            output->parents[1]->grad += output->grad;
            break;
        case Operation::subtract:
            output->parents[0]->grad += output->grad;
            output->parents[1]->grad -= output->grad;
            break;
        case Operation::multiply:
            output->parents[0]->grad += output->grad * output->parents[1]->data;
            output->parents[1]->grad += output->grad * output->parents[0]->data;
            break;
        }
    }
}

void Value::step(double learning_rate)
{
    Tensor scale(
        node_->data.shape(),
        std::vector<double>(node_->data.numel(), learning_rate));
    node_->data -= node_->grad * scale;
}

void Value::zero_grad()
{
    node_->grad = 0.0;
}

void Value::build_topology(
    Node *node,
    std::unordered_set<Node *> &visited,
    std::vector<Node *> &topology)
{
    if (visited.find(node) != visited.end())
    {
        return;
    }
    visited.insert(node);
    for (const auto &parent : node->parents)
    {
        build_topology(parent.get(), visited, topology);
    }
    topology.push_back(node);
}

Value operator+(
    const Value& left,
    const Value& right
){
    auto node = std::make_shared<Node>(
        left.data() + right.data()
    );
    node->opereation = Operation::add;
    node->parents = {left.node_, right.node_};

    return Value(std::move(node));
}
Value operator*(
    const Value& left,
    const Value& right
){
    auto node = std::make_shared<Node>(
        left.data() * right.data()
    );
    node->opereation = Operation::multiply;
    node->parents = {left.node_, right.node_};

    return Value(std::move(node));
}

Value operator-(
    const Value& left,
    const Value& right
){
    auto node = std::make_shared<Node>(
        left.data() - right.data()
    );
    node->opereation = Operation::subtract;
    node->parents = {left.node_, right.node_};

    return Value(std::move(node));
}