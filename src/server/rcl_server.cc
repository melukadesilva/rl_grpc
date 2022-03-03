
#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "../godotInterface/cGodotSharedInterface.h"

#include "../cmake/build/observation_action.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using rcl::RLC;
using rcl::ObservationData;
using rcl::ActionData;

// server logics
class RCServiceImpl final : public RLC::Service {
    
    cGetSharedMemoryTensor mem = cGetSharedMemoryTensor("trainer");
    cSharedMemorySemaphore sem_obs = cSharedMemorySemaphore("trainer_obs_semaphore");
    cSharedMemorySemaphore sem_act = cSharedMemorySemaphore("trainer_act_semaphore");

    cPersistentFloatTensor *obs = mem.findFloatTensor("trainer_observation");
    cPersistentIntTensor *action = mem.findIntTensor("trainer_action");

    Status GetAction(ServerContext *context, 
                     const ObservationData *request, 
                     ActionData *reply) override {

        // https://www.programmersought.com/article/84463145457/
        // Decision *rl_decision = new Decision();
        // int action = 5;

        //rl_decision->set_action_index(action);
        //rl_decision->set_is_end(true);
        std::vector<float> feature_vec;

        for (int i = 0; i < request->features_size(); i++) {
            std::cout << request->features(i) << std::endl;
            feature_vec.push_back(request->features(i));
        }
        // write the observation to shared mem and increment the semaphore
        obs->write(feature_vec);
        sem_obs.post();

        sem_act.wait();
        std::vector<int64_t> action_vec = action->read();
        float action =  action_vec[0];

        if (request->count() == 10) {
            reply->set_is_end(true);
        }
        else {
            reply->set_is_end(false);
        }

        reply->set_action_index(action);// rl_return->set_action_index(action);

        return Status::OK;
    }
};

// run the server
void RunServer() {
    std::string server_address("0.0.0.0:50051");
    RCServiceImpl service;

    //grpc::EnableDefaultHealthCheckService(true);
    //grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    // server builder
    ServerBuilder builder;
    // listen to the given address
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // finally assemble the server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on: " << server_address << std::endl;
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();

}

int main(int argc, char** argv) {
    RunServer();

    return 0;
}