/*
 In computation graph, each node will need:
- its value
- op that created it
- its parent
- its gradient
*/
#include<memory>
#include<vector>
#include<utility>
#include <unordered_set>
#include <algorithm>
#include <assert.h>
#include <iostream>

enum class Operation {
    leaf,
    add,
    subtract,
    multiply
};


struct Node {
    double data = 0.0; // result of forward computation
    double grad = 0.0; // relative change in loss wrt current node
    Operation opereation = Operation::leaf;
    std::vector<std::shared_ptr<Node>> parents; // inputs that produced this node

    explicit Node(double value): data(value){}
};

class Value{
    public:
    Value(double data):node_(std::make_shared<Node>(data)){}
    [[nodiscard]] double data() const noexcept{
        return node_->data;
    }
    [[nodiscard]] double grad() const noexcept{
        return node_->grad;
    }

    friend Value operator+(
        const Value& left,
        const Value& right
    );
        friend Value operator-(
        const Value& left,
        const Value& right
    );
        friend Value operator*(
        const Value& left,
        const Value& right
    );

    void backward(){
        std::unordered_set<Node*> visited;
        std::vector<Node*> topology;
        build_topology(node_.get(), visited, topology);

        node_->grad = 1.0;
        std::reverse(topology.begin(), topology.end());
        for(auto curr = topology.begin(); curr != topology.end(); curr++){
            Node* output = *curr;
            switch(output->opereation){
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
                    output->parents[0]->grad += output->grad*node_->parents[1]->data;  
                    output->parents[1]->grad += output->grad*node_->parents[0]->data;  
                    break;
            }
        }
    }

    private:
    std::shared_ptr<Node> node_;
    explicit Value(std::shared_ptr<Node> node):node_(std::move(node)){};

    static void build_topology(
        Node* node,
        std::unordered_set<Node*>& visited,
        std::vector<Node*>& topology
    ){
        if(visited.find(node) != visited.end()){
            return;
        }
        visited.insert(node);
        for(const auto& p: node->parents){
            build_topology(p.get(), visited, topology);
        }
        topology.push_back(node);
    }
};

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


int main(){
    const Value scalar_prediction(3.5);
    const Value scalar_target(2.5);
    const Value scalar_residual = scalar_prediction - scalar_target;
    
    Value loss = scalar_residual * scalar_residual;
    loss.backward(); 

    assert(loss.grad() == 1.0);
    assert(scalar_residual.grad() == 2.0);
    assert(scalar_prediction.grad() == 2.0);
    assert(scalar_target.grad() == -2.0);

    std::cout << "SUCCESS! \n"; 
}