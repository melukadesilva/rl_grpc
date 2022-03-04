#include <random>

#include <torch/torch.h>

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 2));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


class agentTrainer {
    public:
        struct step_state {
            torch::Tensor action;
            torch::Tensor done_reward;
            torch::Tensor reward;
            torch::Tensor next_observation;
            bool is_done;    
        };

        agentTrainer(float init_epsilon, int num_actions) {
            epsilon = init_epsilon;
            model = std::shared_ptr<Net>();

            // https://www.delftstack.com/howto/cpp/generate-random-number-in-range-cpp/
            std::random_device rd; 
            eng = std::default_random_engine(rd()); // seed the engine with hardware base non-deterministic values
            distrib = std::uniform_int_distribution<int>(0, num_actions);
        };

        torch::Tensor play_step(torch::Tensor observation) {
            torch::Tensor action;

            if (rand() > epsilon) {
                torch::Tensor action_probs = model->forward(observation);
                action = action_probs.argmax(1);// torch::kMax(action_probs, 1);
            }
            else {
                action = torch::tensor(distrib(eng));
            }

            return action;
        };

    private:
        std::shared_ptr<Net> model;
        float epsilon;
        std::uniform_int_distribution<int> distrib;
        std::default_random_engine eng;
};