# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging
from urllib import response

import grpc
import observation_action_pb2
import observation_action_pb2_grpc

import torch
import gym

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

from InvertedPendulumEnv import InvPendulumEnv


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # while True:
    with grpc.insecure_channel('localhost:50051') as channel:
        # create the env
        env = InvPendulumEnv(channel)
        # Instantiate the agent
        model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log="./ddpg_try_1")
        # Train the agent
        model.learn(total_timesteps=1_000)#int(2e5))

        # Save the agent and clear the memory
        model.save("./checkpoints/ddpg_inv_pendulum")
        del model

        # Load and evaluate the model
        model = DDPG.load("./checkpoints/ddpg_inv_pendulum", env=env)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

        print("Mean reward: {}, Standard deviation reward: {}".format(mean_reward, std_reward))

        env.terminate()
        '''
        while True:
            obs, rew, done, _ = env.step(torch.tensor([8.0]))
            print(rew, obs, done)

            if done == 1:
                response_reset = env.reset()
        '''

if __name__ == '__main__':
    logging.basicConfig()
    run()
