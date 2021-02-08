import os
import numpy as np
import logging
import PIL

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from tensorflow.keras.layers import Lambda
from tensorflow.compat.v1.train import RMSPropOptimizer
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Debugging Information
deprecation._PRINT_DEPRECATION_WARNINGS = False # Disable Deprecation Warnings

class BreakoutV4:
    def __init__(self, max_episode_steps, update_period, preprocessing):
        self.max_episode_steps = max_episode_steps
        self.update_period = update_period
        self.preprocessing = preprocessing

    def create_env(self):
        env = suite_atari.load(
            "BreakoutNoFrameskip-v4",
            max_episode_steps = self.max_episode_steps,
            gym_env_wrappers = self.preprocessing)

        tf_env = TFPyEnvironment(env)
        return tf_env

    def DQN(self, tf_env):
        preprocessing_layer = Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
        conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
        fc_layer_params = [512]

        q_net = QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            preprocessing_layers = preprocessing_layer,
            conv_layer_params = conv_layer_params,
            fc_layer_params = fc_layer_params)
        
        return q_net

    def agent(self, tf_env, q_net, target_update_period, gamma):
        train_step = tf.Variable(0)
        
        # Hyperparameters As Used In [1]
        # [1]:  https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL
        optimizer = RMSPropOptimizer(
            learning_rate = 2.5e-4, decay = 0.95, momentum = 0.0,
            epsilon = 1e-5, centered = True)
        
        epsilon_fn = PolynomialDecay(
            initial_learning_rate = 1.0,
            decay_steps = 250000 // self.update_period, 
            end_learning_rate = 0.01) 

        agent = DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network = q_net, 
            optimizer = optimizer,
            target_update_period = target_update_period,
            td_errors_loss_fn = Huber(reduction = "none"), # Return Error Per Instance (Not Mean)
            gamma = gamma, 
            train_step_counter = train_step,
            epsilon_greedy = lambda: epsilon_fn(train_step)) # Pass Lambda As Expects No Argument

        return agent
    
    def replay_buffer(self, tf_env, agent, max_length):

        replay_buffer = TFUniformReplayBuffer(
            data_spec = agent.collect_data_spec,
            batch_size = tf_env.batch_size,
            max_length = max_length)   

        replay_buffer_observer = replay_buffer.add_batch
        return replay_buffer, replay_buffer_observer
    
    class ShowProgress:
        def __init__(self, total):
            self.counter = 0
            self.total = total

        def __call__(self, trajectory):
            if not trajectory.is_boundary():
                self.counter += 1
            if self.counter % 100 == 0:
                print("\r{}/{}".format(self.counter, self.total), end = "")

    def train_metrics(self):
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        logging.getLogger().setLevel(logging.INFO)
        log_metrics(train_metrics)

        return train_metrics

    def collect_driver(self, tf_env, agent, replay_buffer_observer, train_metrics):
        collect_driver = DynamicStepDriver(
            tf_env,
            agent.collect_policy,
            observers = [replay_buffer_observer] + train_metrics,
            num_steps = self.update_period)  

        return collect_driver  
    
        
    def dataset(self, replay_buffer, sample_batch_size, num_steps, num_parallel_calls):
        dataset = replay_buffer.as_dataset(
            sample_batch_size = sample_batch_size,
            num_steps = num_steps,
            num_parallel_calls = num_parallel_calls).prefetch(3)

        return dataset

    def train_agent(self, tf_env, collect_driver, agent, dataset, n_iterations, train_metrics):
        collect_driver.run = function(collect_driver.run)
        agent.train = function(agent.train)

        time_step = None
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        iterator = iter(dataset)
        for iteration in range(n_iterations):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            print("\r{} Loss: {:.5f}, ".format(iteration, train_loss.loss.numpy()), end = "")
            if iteration % 1000 == 0:
                log_metrics(train_metrics)

        return agent