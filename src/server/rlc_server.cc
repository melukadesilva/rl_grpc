
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

using rlc::RLC;
using rlc::ObservationData;
using rlc::ActionData;
using rlc::Empty;

// server logics
class RCServiceImpl final : public RLC::Service {
    
    // make a cSharedMemoryTensor
    cSharedMemoryTensor mem = cSharedMemoryTensor("env");
    cSharedMemorySemaphore sem_obs = cSharedMemorySemaphore("obs_semaphore", 0);
    cSharedMemorySemaphore sem_act = cSharedMemorySemaphore("act_semaphore", 0);

    // write to the shared memory
    //cPersistentIntTensor *actionTensor = mem.newIntTensor("action", 1);
    cPersistentFloatTensor *actionTensor = mem.newFloatTensor("action", 1);
    cPersistentIntTensor *envActionTensor = mem.newIntTensor("env_action", 2);
    cPersistentFloatTensor *observationTensor = mem.newFloatTensor("observation", 3);
    cPersistentFloatTensor *rewardTensor = mem.newFloatTensor("reward", 1);
    cPersistentIntTensor *doneTensor = mem.newIntTensor("done", 1);

    /*
    std::vector<int64_t> env_action_vec;
    std::vector<float> observations, action_vec;
    float reward;
    int64_t is_done;
    */
    
    Status Step(ServerContext *context, 
                     const ActionData *request, 
                     ObservationData *reply) override {

        // https://www.programmersought.com/article/84463145457/
        // Decision *rl_decision = new Decision();
        // int action = 5;

        //rl_decision->set_action_index(action);
        //rl_decision->set_is_end(true);

        std::vector<int64_t> env_action_vec;
        std::vector<float> observations, action_vec;
        float reward;
        int64_t is_done;

        // std::cout << request->action_index() << std::endl;
        action_vec.push_back(request->action_index());
        actionTensor->write(action_vec);
        env_action_vec.push_back(request->env_action());
        envActionTensor->write(env_action_vec);
        sem_act.post();

        sem_obs.wait();
        observations = observationTensor->read();
        reward = rewardTensor->read()[0];
        is_done = doneTensor->read()[0];
        
        //if (is_done == 1) {
            //std::cout << is_done << std::endl;
        reply->set_is_done(is_done);
        //}
        //else {
        //    reply->set_is_done(0);
        //}
        
        //reply->set_features(observations);
        for(int i = 0; i < observations.size(); ++i) {
            reply->add_observations(observations[i]);
        }
        reply->set_reward(reward);
        

        return Status::OK;
    }

    Status Reset(ServerContext *context,
                const Empty *request,
                ObservationData *reply) override {

        std::vector<float> observations;
        float reward;
        int64_t is_done;
        
        std::vector<int64_t> reset = {1, 0};
        // std::cout << "reset: " << reset[0] << std::endl;
        envActionTensor->write(reset);
        sem_act.post();

        sem_obs.wait();
        observations = observationTensor->read();
        reward = rewardTensor->read()[0];
        is_done = doneTensor->read()[0];

        //if (is_done == 1) {
        // std::cout << "is done: " << is_done << std::endl;
        reply->set_is_done(is_done);
        //}
        //else {
        //    reply->set_is_done(0);
        //}
        
        //reply->set_features(observations);
        for(int i = 0; i < observations.size(); ++i) {
            reply->add_observations(observations[i]);
        }
        reply->set_reward(reward);
        
        return Status::OK;
    }

    Status Terminate(ServerContext *context,
                const Empty *request,
                Empty *reply) override {
            
        std::vector<int64_t> terminate = {0, 1};
        // std::cout << "reset: " << reset[0] << std::endl;
        envActionTensor->write(terminate);
        sem_act.post();

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