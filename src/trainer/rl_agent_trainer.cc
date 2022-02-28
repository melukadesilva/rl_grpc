#include "../godotInterfaceTorch/cGodotSharedInterface.h"

int main(int argc, char* argv[]) {
    // make a cSharedMemoryTensor
    cSharedMemoryTensor mem("trainer");
    cSharedMemorySemaphore sem_obs("trainer_obs_semaphore", 0);
    cSharedMemorySemaphore sem_act("trainer_act_semaphore", 0);

    torch::Tensor test_tensor = torch::ones(3);
    // std::cout << test_tensor << std::endl;

    // write to the shared memory
    cPersistentFloatTensor *action = mem.newFloatTensor("trainer_action", 3);
    cPersistentFloatTensor *observation = mem.newFloatTensor("trainer_observation", 3);

    for(;;) {
        sem_obs.wait();
        torch::Tensor obs = observation->read();
        std::cout << "observation received " << obs  << std::endl;

        action->write(test_tensor);
        sem_act.post();
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