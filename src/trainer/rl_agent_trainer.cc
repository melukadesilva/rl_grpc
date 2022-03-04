#include "../godotInterfaceTorch/cGodotSharedInterface.h"
#include "agent.h"
#include "replayBuffer.h"

// define a nn by inheriting from torch::nn::Module
struct MLPNet : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    MLPNet() {
        // construct and register the linear modules
        fc1 = register_module("fc1", torch::nn::Linear(3, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 2));
    }

    // Implement the algorithm
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc2->forward(x));
        //x = torch::log_softmax(fc3->forward(x), 1);
        x = torch::softmax(fc3->forward(x), 1);
        return x;
    }
};

int main(int argc, char* argv[]) {
    // create the new network
    auto net = std::make_shared<MLPNet>();
    float init_epsilon = 0.01;
    int num_actions = 2;
    agentTrainer policy_agent = agentTrainer(init_epsilon, num_actions);

    int buffer_size = 8;
    int batch_size = 2;
    replayBuffer experianceBuffer = replayBuffer(buffer_size, batch_size);

    // make a cSharedMemoryTensor
    cSharedMemoryTensor mem("trainer");
    cSharedMemorySemaphore sem_obs("trainer_obs_semaphore", 0);
    cSharedMemorySemaphore sem_act("trainer_act_semaphore", 0);

    // torch::Tensor test_tensor = torch::rand(1); //torch::ones(3);
    // std::cout << test_tensor << std::endl;

    // write to the shared memory
    cPersistentIntTensor *action = mem.newIntTensor("trainer_action", 1);
    cPersistentFloatTensor *observation = mem.newFloatTensor("trainer_observation", 3);
    cPersistentFloatTensor *reward = mem.newFloatTensor("trainer_reward", 1);
    cPersistentIntTensor *done = mem.newIntTensor("trainer_done", 1);

    int count = 0;
    torch::Tensor prev_observation;
    torch::Tensor prev_action;

    for(;;) {
        //torch::Tensor test_tensor = torch::randn(1); 
        sem_obs.wait();
        torch::Tensor obs = observation->read();
        torch::Tensor rwd = reward->read();
        torch::Tensor is_done = done->read();

        torch::Tensor obs_rs = obs.resize_({1, 3});

        torch::Tensor action_probs = net->forward(obs_rs);
        //torch::Tensor action_prediction = policy_agent.play_step(obs_rs);

        torch::Tensor action_prediction = torch::multinomial(action_probs, 1, true);

        //torch::Tensor action_prediction = torch::_cast_Long(torch::ones(1) * 50);

        //std::cout << "observation received " << obs_rs  << std::endl;
        std::cout << "Predicted action " << action_prediction  << std::endl;

        action->write(action_prediction);
        sem_act.post();

        // update the transitions accordingly
        if (count > 0) {
            replayBuffer::transition tr = {
                prev_observation,
                prev_action,
                obs_rs,
                rwd,
                is_done,
            };
        }

        prev_observation = obs_rs;
        prev_action = action_prediction;
    }


    // read the tensor
    // torch::Tensor a = tens->read();
    // std::cout << a << std::endl;

    /*
    managed_shared_memory segment(open_only, "test_segment");

    FloatVector *v = segment.find<FloatVector> ("T").first;
    std::cout<<"C++ from Python = [";
    for(int i=0;i<v->size(); i++){
        std::cout<<(*v)[i]<<", ";
    }
    std::cout<<"]";
    */

    return 0;
}