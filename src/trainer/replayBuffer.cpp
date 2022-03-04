#include "replayBuffer.h"

//test
int main() {
    int batch_size = 5;
    int buffer_size = 2;

    int observation = 5;
    int action = 1;
    int next_observation = 3;
    int reward = 10;

    replayBuffer buffer = replayBuffer(buffer_size, batch_size);
    
    //boost::circular_buffer<replayBuffer::transition> cBuffer = buffer.init();
    replayBuffer::transition tr1 = {observation, action, next_observation, reward};

    buffer.push(tr1);

    int observation_1 = 5;
    int action_1 = 1;
    int next_observation_1 = 3;
    int reward_1 = 10;

    replayBuffer::transition tr2 = {observation_1, action_1, next_observation_1, reward_1};

    buffer.push(tr2);   

    //cBuffer.push_back(tr1);
    //cBuffer.push_back(tr2); 

    std::vector<replayBuffer::transition> tr = buffer.sample();


    std::cout << tr[0].action << std::endl;

}