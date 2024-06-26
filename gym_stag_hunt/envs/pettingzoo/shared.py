from pettingzoo.utils import wrappers
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
import functools
import time


def default_wrappers(env_init):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


class PettingZooEnv(ParallelEnv):
    def __init__(self, og_env):
        super().__init__()

        self.env = og_env

        self.possible_agents = ["player_" + str(n) for n in range(2)]
        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)

        self._action_spaces = {
            agent: self.env.action_space for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: self.env.observation_space for agent in self.possible_agents
        }

        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        if self.env.end_ep_on_reward:
            self.dones["__all__"] = False
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_observations = {
            agent: self.env.observation_space.sample() for agent in self.agents
        }
        self.t = 0
        self.last_rewards = [0.0, 0.0]

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.env.action_space

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        if self.env.end_ep_on_reward:
            self.dones["__all__"]=False
        obs = self.env.reset()
        self.accumulated_actions = []
        self.current_observations = {agent: obs for agent in self.agents}
        self.t = 0

        return self.current_observations

    
    def step(self, actions):
        """Calling step in gym/abstract_markov_staghunt.py which calls 
         update in games/staghunt_game.py """
        
        step_observations, step_rewards, step_dones, info = self.env.step(actions)

        # TODO add config flag for env rendering during training
        # self.env.render(mode='human')
        # time.sleep(0.5)

        if self.env.end_ep_on_reward:
            # Initialize containers for this step's data
            obs = {}
            rewards = {}
            infos = {}
            local_dones = {}

            for agent_id in list(self.agents):
                agent_idx = int(str(agent_id)[-1])
                # Update only if agent is active at the start of this step
                if not self.dones[agent_id]:
                    obs[agent_id] = step_observations[agent_idx] 
                    rewards[agent_id] = step_rewards[agent_idx]
                    infos[agent_id] = info
                    self.dones[agent_id] = step_dones[agent_idx]
                    local_dones[agent_id] = step_dones[agent_idx]
                    if step_dones[agent_idx]:
                        self.agents.remove(agent_id)
                    
            local_dones["__all__"] = step_dones[-1]
            self.dones["__all__"] = step_dones[-1]
            return obs, rewards, local_dones, infos
        else:
            obs = {self.agents[0]: step_observations[0], self.agents[1]: step_observations[1]}
            rewards = {self.agents[0]: step_rewards[0], self.agents[1]: step_rewards[1]}
            local_dones = {agent: False for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
            return obs, rewards, local_dones, infos
    
    def observe(self, agent):
        return self.current_observations[agent]

    def state(self):
        pass
