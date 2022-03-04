#include <iostream>
#include <vector>
#include <random> // using c11 random number generator

#include <boost/circular_buffer.hpp>
#include <torch/torch.h>

//#include <torch/torch.h>

class replayBuffer {
    public:
        /*
        struct transition {
                int observation;
                int action;
                int next_observation;
                int reward;
        };
        */
        struct transition {
            torch::Tensor observations;
            torch::Tensor action;
            torch::Tensor next_observation;
            torch::Tensor reward;
            torch::Tensor done;
        };
    
        // constructor initialise the circular buffer        
        replayBuffer(int buffer_size, int batch_size) {
            buf_size = buffer_size;
            bat_size = batch_size;

            //boost::circular_buffer<transition> cBuffer(batch_size);
            cBuffer = boost::circular_buffer<transition>(buf_size);
        };

        // insert a transition to the circular buffer
        void push(transition tr) {
            cBuffer.push_back(tr);
        }

        // sample a random batch of samples from the circular buffer
        std::vector<transition> sample() {
            // initialise the random number generation
            // https://www.delftstack.com/howto/cpp/generate-random-number-in-range-cpp/
            std::random_device rd; 
            std::default_random_engine eng(rd()); // seed the engine with hardware base non-deterministic values
            std::uniform_int_distribution<int> distrib(0, bat_size);

            std::vector<transition> transition_batch;
            // sample a random int, index the buffer and store it in transition vector to return
            for (int i = 0; i <  bat_size; i++) {
                int index = distrib(eng);
                transition sampled_transition = cBuffer[index];
                transition_batch.push_back(sampled_transition);
            }
            // make a transition batch and return
            std::cout << cBuffer.capacity() << std::endl;
            std::cout << cBuffer.size() << std::endl;

            return transition_batch;
        };

    private:
    /*
        struct transition {
            torch::Tensor observations;
            torch::Tensor action;
            torch::Tensor next_observation;
            torch::Tensor reward;
        }
    */
        // replay buffer size
        int buf_size;
        // batch size to sample from the buffer
        int bat_size;
        // a circular buffer to store the transitions
        boost::circular_buffer<transition> cBuffer;
};