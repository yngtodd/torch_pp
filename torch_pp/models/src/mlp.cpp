#include <mlp.hpp>

namespace mlp {

    auto create_mlp() 
    {
        auto net = std::make_shared<mlp::MLP>();
        return net;
    }

}
