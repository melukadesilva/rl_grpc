
#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "../godotInterface/cGodotSharedInterface.h"
//#include <godotInterface/cGodotSharedInterface.h>
// #include <boost/interprocess/managed_shared_memory.hpp>

#include "../cmake/build/observation_action.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using rcl::RLC;
using rcl::ObservationData;
using rcl::ActionData;

class RLCClient {
    public:
        RLCClient(std::shared_ptr<Channel> channel) : stub_(RLC::NewStub(channel)) {};
        
        ActionData GetAction(std::vector<float> data, float reward, int done) {
            ObservationData request;
            for (int i=0; i < data.size(); i++) {
                request.add_features(data[i]);
            }

            request.set_reward(reward);
            request.set_done(done);

            ActionData reply;

            ClientContext context;

            Status status = stub_->GetAction(&context, request, &reply);

            if (status.ok()) {
                return reply;
            }
            else {
                ActionData failed_action;
                failed_action.set_action_index(-1);
                failed_action.set_is_end(true);

                std::cout << status.error_code() << ":" << status.error_message() << std::endl;
                return failed_action;
            }
        }
    private:
        std::unique_ptr<RLC::Stub> stub_;
};

int main(int argc, char** argv) {
    std::string target_str = "0.0.0.0:50051";
    RLCClient rcl(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    std::cout << "listening at: " << target_str << std::endl;
    
    // make a cSharedMemoryTensor
    cSharedMemoryTensor mem("env");
    cSharedMemorySemaphore sem_obs("obs_semaphore", 0);
    cSharedMemorySemaphore sem_act("act_semaphore", 0);

    // write to the shared memory
    cPersistentIntTensor *action = mem.newIntTensor("action", 1);
    cPersistentFloatTensor *observation = mem.newFloatTensor("observation", 3);
    cPersistentFloatTensor *reward = mem.newFloatTensor("reward", 1);
    cPersistentIntTensor *done = mem.newIntTensor("done", 1);

    // std::vector<float> obs {1.4, 2.2, 3.1, 4.5};
    int count = 0;

    for (;;) {
        sem_obs.wait();
        std::vector<float> obs = observation->read();
        std::vector<float> rwd = reward->read();
        std::vector<int64_t> is_done = done->read(); 
        //std::cout << "observation received " << obs  << std::endl;

        ActionData data = rcl.GetAction(obs, rwd[0], is_done[0]);

        //int current_action = data.action_index();
        //std::cout << "recieved action " << current_action << std::endl;
        std::vector<int64_t> current_action_vec;
        current_action_vec.push_back(data.action_index());
        
        for (int i = 0; i < current_action_vec.size(); i++) {
            std::cout << "recieved action " << current_action_vec[i] << std::endl;
        }
        
        //std::cout << "recieved action " << current_action[0] << std::endl;

        action->write(current_action_vec);
        sem_act.post();

        if (data.is_end() == true) {
            std::cout << "is the end " << data.is_end() << std::endl;
            std::cout << "count is: " << count << std::endl;
            //break;
        }
        count++;
    }

    return 0;
}