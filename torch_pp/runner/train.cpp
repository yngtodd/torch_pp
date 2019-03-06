#include <iostream>
#include <torch/torch.h>
#include <mlp.hpp>
#include <mnist.hpp>

int main() {
    std::cout << "Hello World!" << std::endl;
    //auto net = std::make_shared<mlp::MLP>();
    mlp::MLP net;

    /*
     * for (const auto& pair : net.named_parameters()) {
     *   std::cout << pair.key() << ": " << pair.value() << std::endl;
     * }
     */

    std::cout << net.forward(torch::ones({2, 784})) << std::endl;
}
