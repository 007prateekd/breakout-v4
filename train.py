from agent import *
import os
import tensorflow.python.util.deprecation as deprecation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable Debugging Information
deprecation._PRINT_DEPRECATION_WARNINGS = False # Disable Deprecation Warnings

init = BreakoutV4(
    max_episode_steps = 27000, update_period = 4,
    preprocessing = [AtariPreprocessing, FrameStack4]
)

tf_env = init.create_env()
q_net = init.DQN(tf_env)
agent = init.agent(tf_env, q_net, 2000, 0.99)
replay_buffer, replay_buffer_observer = init.replay_buffer(tf_env, agent, 10 ** 5)
train_metrics = init.train_metrics()
collect_driver = init.collect_driver(tf_env, agent, replay_buffer_observer, train_metrics)

# Warming Up Buffer
print("\nWarming Up Buffer:")
initial_collect_policy = RandomTFPolicy(
    tf_env.time_step_spec(),
    tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers = [replay_buffer.add_batch, init.ShowProgress(10000)],
    num_steps = 10000) 

final_time_step, final_policy_state = init_driver.run()

dataset = init.dataset(replay_buffer, 64, 2, 3)
print("\nTraining Agent:")
agent = init.train_agent(tf_env, collect_driver, agent, dataset, 10 ** 3, train_metrics)