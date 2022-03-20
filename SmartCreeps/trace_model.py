from stable_baselines3 import DDPG, PPO
import torch
import torch.nn as nn

model = PPO.load("./best_model")
#model = DDPG.load("./ddpg_mars")
print(model.policy.mlp_extractor.policy_net)
print(model.policy.action_net)
model = nn.Sequential(
	model.policy.mlp_extractor.policy_net,
	model.policy.action_net
)

#print(model.policy.actor)
#model = model.policy.actor
#print(model.policy.critic)
#model.policy.actor.eval()
inp = torch.rand(1, 8)

traced_script_module = torch.jit.trace(model, inp)
traced_script_module.save("ppo_actor.jit")
