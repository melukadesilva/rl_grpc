extends Node


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
var mob
var mob_pos

var player
var player_pos

var distant
var distant_reward = 0.0
var hit_reward = 0.0
var total_reward

var is_done = 0
var timeout = true
var deltat = 0.05

var pos_obs_x
var pos_obs_y
var velo_obs_x
var velo_obs_y
var observations

var mem
var sem_action
var sem_observation
var agent_action_tensor
var env_action_tensor
var reward_tensor
var observation_tensor
var done_tensor

var action
var env_action

var train = true
var time_elapsed = 0.0
var episode_length = 10.0

func apply_action(action):
	mob.velocity = Vector2(action[0], action[1])
	mob.direction = action[2]

func get_distance_reward():
	player_pos = player.position
	mob_pos = mob.position
	
	distant = sqrt(pow((player_pos.x - mob_pos.x), 2) + pow((player_pos.y - mob_pos.y), 2))
	distant_reward += 1 / distant
	distant_reward = distant_reward - time_elapsed * 1e-2

func reset():
	print("Resetting")
	mob.reset()
	player.reset()
	
	hit_reward = 0.0
	distant_reward = 0.0
	is_done = 0

# Called when the node enters the scene tree for the first time.
func _ready():
	randomize()
	mob = $Mob
	player = $Player
	
	mem = cSharedMemory.new()
	mem.init("env")
	if train:
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()

		sem_action.init("act_semaphore")
		sem_observation.init("obs_semaphore")
		
		#agent_action_tensor = mem.findIntTensor("action")
		agent_action_tensor = mem.findFloatTensor("action")
		env_action_tensor = mem.findIntTensor("env_action")
		reward_tensor = mem.findFloatTensor("reward")
		observation_tensor = mem.findFloatTensor("observation")
		done_tensor = mem.findIntTensor("done")
		print("Running as OpenAIGym environment")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if timeout:
		if train:
			sem_action.wait()
			action = agent_action_tensor.read()
			env_action = env_action_tensor.read()
			
		apply_action(action)
		# print(action)
		
		get_distance_reward()
		
		if env_action[0] == 1:
			reset()
			time_elapsed = 0.0
		if env_action[1] == 1:
			get_tree().quit()
		
		$Timer.start(deltat)
		timeout = false
	
func game_over():
	hit_reward = 100.0
	is_done = 1
	# get_tree().paused = true

func _on_Timer_timeout():
	total_reward = hit_reward + distant_reward
	
	pos_obs_x = abs(player_pos.x - mob_pos.x)
	pos_obs_y = abs(player_pos.y - mob_pos.y)
	velo_obs_x = abs(player.velocity.y - mob.velocity.y)
	velo_obs_y = abs(player.velocity.y - mob.velocity.y)
	observations = [pos_obs_x, pos_obs_y, velo_obs_x, velo_obs_y]
	
	# print(total_reward)
	# print(pos_obs_x)
	# print(pos_obs_y)
	
	if train:
		observation_tensor.write(observations)
		reward_tensor.write([total_reward])
		done_tensor.write([is_done])
		
		sem_observation.post()
	
	time_elapsed += deltat
	timeout = true
	
	# print(time_elapsed)
	
	
	
	
