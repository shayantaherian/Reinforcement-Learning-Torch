import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import wrappers
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()
class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class Agent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, alpha, beta, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                     layer2_size, n_actions=1)
        self.log_probs = None

    def choose_action(self, observation):
        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

class NewAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()

if __name__ == '__main__':
    agent = NewAgent(alpha=0.00001, input_dims=[8], gamma=0.99,
                  n_actions=4, layer1_size=2048, layer2_size=512)

    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 1000
    for i in range(num_episodes):

        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward


        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar('average score', avg_score, i)
        print('episode: ', i,'score: %.2f' % score)