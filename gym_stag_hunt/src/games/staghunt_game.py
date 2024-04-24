from numpy import zeros, uint8, array, hypot, empty

from gym_stag_hunt.src.games.abstract_grid_game import AbstractGridGame

from gym_stag_hunt.src.utils import (
    overlaps_entity,
    place_entity_in_unoccupied_cell,
    spawn_plants,
    respawn_plants,
)

# Entity Keys
A_AGENT = 0
B_AGENT = 1
STAG = 2
PLANT = 3


class StagHunt(AbstractGridGame):
    def __init__(
        self,
        stag_reward,
        stag_follows,
        stag_frozen,
        stag_random_respawn,
        agent_random_respawn,
        run_away_after_maul,
        opponent_policy,
        forage_quantity,
        forage_reward,
        mauling_punishment,
        timestep_penalty,
        end_ep_on_reward,
        no_plants,
        done_bits,
        # Super Class Params
        window_title,
        grid_size,
        screen_size,
        obs_type,
        load_renderer,
        enable_multiagent,
    ):
        """
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """

        super(StagHunt, self).__init__(
            grid_size=grid_size,
            screen_size=screen_size,
            obs_type=obs_type,
            enable_multiagent=enable_multiagent,
        )

        # Config
        self._stag_follows = stag_follows
        self._stag_frozen = stag_frozen
        self._stag_random_respawn = stag_random_respawn
        self._agent_random_respawn = agent_random_respawn
        self._run_away_after_maul = run_away_after_maul
        self._opponent_policy = opponent_policy
        self._end_ep_on_reward = end_ep_on_reward
        self._done_bits = done_bits
        self._obs_type = obs_type

        # Reinforcement Variables
        self._stag_reward = stag_reward  # record RL values as attributes
        self._forage_quantity = forage_quantity
        self._forage_reward = forage_reward
        self._mauling_punishment = mauling_punishment
        self._timestep_penalty = timestep_penalty

        # State Variables
        self._tagged_plants = []  # harvested plants that need to be re-spawned
        self._playerA_done = False
        self._playerB_done = False
        self._stag_done = False

        # Entity Positions
        self._stag_pos = zeros(2, dtype=uint8)
        self._plants_pos = []
        self._no_plants = no_plants
        self.reset_entities()  # place the entities on the grid

        # If rendering is enabled, we will instantiate the rendering pipeline
        if obs_type == "image" or load_renderer:
            # we don't want to import pygame if we aren't going to use it, so that's why this import is here
            from gym_stag_hunt.src.renderers.hunt_renderer import HuntRenderer

            self._renderer = HuntRenderer(
                game=self, window_title=window_title, screen_size=screen_size
            )

    """
    Collision Logic
    """

    def _overlaps_plants(self, a, plants):
        """
        :param a: (X, Y) tuple for entity 1
        :param plants: Array of (X, Y) tuples corresponding to plant positions
        :return: True if a overlaps any of the plants, False otherwise
        """
        if self._no_plants:
            return False
        for x in range(0, len(plants)):
            pos = plants[x]
            if a[0] == pos[0] and a[1] == pos[1]:
                # Only tag plants in fixed horizon envs where plant respawn is necessary
                if not self._end_ep_on_reward:
                    self._tagged_plants.append(x)
                return True
        return False

    """
    State Updating Methods
    """

    def _calc_reward(self):
        """
        Calculates the reinforcement rewards for the two agents.
        :return: A tuple R where R[0] is the reinforcement for A_Agent, and R[1] is the reinforcement for B_Agent
        """
        if overlaps_entity(self.A_AGENT, self.STAG):
            if overlaps_entity(self.B_AGENT, self.STAG): 
                # A and B are on stag -> check for successful stag hunt or remaining agent mauled on top of an agent that was already mauled at a previous timestep:
                if self._playerA_done:
                    rewards = (0, self._mauling_punishment)
                elif self._playerB_done:
                    rewards = (self._mauling_punishment, 0)
                else: 
                    rewards = (self._stag_reward, self._stag_reward)  
            else:
                # A is on stag, B is not -> A is mauled, check for B's forage status:
                if self._overlaps_plants(self.B_AGENT, self.PLANTS): 
                    rewards = (self._mauling_punishment, self._forage_reward)  
                else:
                    rewards = (self._mauling_punishment, 0)  

        elif overlaps_entity(self.B_AGENT, self.STAG):
            # B is on stag, A is not -> B is mauled, check for A's forage status:
            if self._overlaps_plants(self.A_AGENT, self.PLANTS):
                rewards = (self._forage_reward, self._mauling_punishment)  
            else:
                rewards = (0, self._mauling_punishment)  
        elif self._overlaps_plants(self.A_AGENT, self.PLANTS):
            # Neither player on stag, A on plant -> check for B's forage status:
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                # A and B are on a plant -> allowing agents to double forage same plant for now 
                rewards = (self._forage_reward, self._forage_reward)
            else:
                # Only A on plant
                rewards = (self._forage_reward, 0)
        else:
            # Neither player on stag, A not on plant -> check for B's forage status:
            if self._overlaps_plants(self.B_AGENT, self.PLANTS):
                rewards = 0, self._forage_reward  
            else:
                rewards = 0, 0  

        # Adjust for timestep penalty and assign reward regardless of inf/fixed horizon
        penalty = self._timestep_penalty if self._timestep_penalty != 0 else 0
        rewardA, rewardB = (float(rewards[0] + penalty), float(rewards[1] + penalty))

        # Removing repeated reward for players on entity if they are done
        if self._end_ep_on_reward:
            if self._playerA_done:
                rewardA = 0
            if self._playerB_done:
                rewardB = 0

        return (rewardA, rewardB)
    

    def update(self, agent_moves):
        """
        Takes in agent actions and calculates next game state.
        :param agent_moves: If multi-agent, a tuple of actions. Otherwise a single action and the opponent takes an
                            action according to its established policy.
        :return: observation, rewards, is the game done
        """

        # Move Entities
        if not self._stag_frozen:
            self._move_stag()
        if self._enable_multiagent:
            self._move_agents(agent_moves=agent_moves)
        else:
            if self._opponent_policy == "random":
                self._move_agents(
                    agent_moves=[agent_moves, self._random_move(self.B_AGENT)]
                )
            elif self._opponent_policy == "pursuit":
                self._move_agents(
                    agent_moves=[
                        agent_moves,
                        self._seek_entity(self.B_AGENT, self.STAG),
                    ]
                )

        # Get Rewards
        true_iteration_rewards = self._calc_reward()
        
        # Pretend timestep penalty doesn't exist to bypass modifying game logic
        reverse_penalty = (-1*self._timestep_penalty) if self._timestep_penalty != 0 else 0
        penalty_adjusted_iteration_rewards = (true_iteration_rewards[0]+reverse_penalty, true_iteration_rewards[1]+reverse_penalty)

        # Stag was hunted
        if penalty_adjusted_iteration_rewards == (self._stag_reward, self._stag_reward):
            if not self._end_ep_on_reward:
                self.STAG = place_entity_in_unoccupied_cell(
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.PLANTS + self.AGENTS + [self.STAG])
            else:
                # Players are done for the timestep and episode
                self._playerA_done, self._playerB_done = True, True

        # One player was mauled
        if (self._mauling_punishment in penalty_adjusted_iteration_rewards):
            if self._end_ep_on_reward:
                # Only want to update done variables if player isn't already 'done'
                self._playerA_done = self._playerA_done or (not self._playerA_done and penalty_adjusted_iteration_rewards[0] == self._mauling_punishment)
                self._playerB_done = self._playerB_done or (not self._playerB_done and penalty_adjusted_iteration_rewards[1] == self._mauling_punishment)
                
            # Reset stag if needed
            if self._run_away_after_maul and not self._end_ep_on_reward: 
                self.STAG = place_entity_in_unoccupied_cell(
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.PLANTS + self.AGENTS + [self.STAG],
                )

        # Atleast one player foraged
        if self._forage_reward in penalty_adjusted_iteration_rewards:
            if not self._end_ep_on_reward:
                # Respawn plants if needed
                new_plants = respawn_plants(
                    plants=self.PLANTS,
                    tagged_plants=self._tagged_plants,
                    grid_dims=self.GRID_DIMENSIONS,
                    used_coordinates=self.AGENTS + [self.STAG])
                self._tagged_plants = []
                self.PLANTS = new_plants
            else:
                # Only want to update done variables if player isn't already 'done'
                self._playerA_done = self._playerA_done or (not self._playerA_done and penalty_adjusted_iteration_rewards[0] == self._forage_reward)
                self._playerB_done = self._playerB_done or (not self._playerB_done and penalty_adjusted_iteration_rewards[1] == self._forage_reward)
    
        info = {}
        obs = self.get_observation()
        dones_all = self._playerA_done and self._playerB_done
        
        if self._enable_multiagent:
            if self._obs_type == "coords":
                return (
                    (obs, self._flip_coord_observation_perspective(obs)),
                    true_iteration_rewards,
                    (self._playerA_done, self._playerB_done, dones_all),
                    info,
                )
            elif self._obs_type == "grid" or self._obs_type == "grid_onehot":
                obs_A, obs_B = self._coord_to_grid_observation()
                return (obs_A, obs_B), true_iteration_rewards, (self._playerA_done, self._playerB_done, dones_all), info
            else:
                return (obs, obs), true_iteration_rewards, (self._playerA_done, self._playerB_done, dones_all), info
        else:
            # TODO modify to make infinite horizon work for single agent case 
            return obs, true_iteration_rewards[0], self._playerA_done, info

    def _coord_observation(self):
        """
        :return: list of all the entity coordinates
        """
        shipback = [self.A_AGENT, self.B_AGENT, self.STAG]
        shipback = shipback + self.PLANTS 
        
        if self._obs_type == 'grid' or self._obs_type == 'grid_onehot':
            grid_A, _ = self._coord_to_grid_observation()
            # Must return single observation for RLLib preprocessor
            return grid_A
        else:
            if self._done_bits:
                done_states = [[1 if self._playerA_done else 0, 1 if self._playerB_done else 0]]
                shipback = shipback + done_states
            return array(shipback).flatten()
        
    def _coord_to_grid_observation(self):  
        shipback = [self.A_AGENT, self.B_AGENT, self.STAG]
        shipback = shipback + self.PLANTS 

        a_agent = 1 << 0  # Bit 0
        a_agent_done = 1 << 1  # Bit 1
        b_agent = 1 << 2  # Bit 2
        b_agent_done = 1 << 3  # Bit 3
        stag = 1 << 4  # Bit 4
        plant = 1 << 5  # Bit 5
        plant_2 = 1 << 6  # Bit 6

        encoding_idx_map = { a_agent:0,
                                a_agent_done:1,  # Bit 1
                                b_agent:2,  # Bit 2
                                b_agent_done:3, # Bit 3
                                stag:4,
                                plant:5,
                                plant_2:6}
        
        grid_A = zeros((5, 5), dtype=uint8)
        grid_B = zeros((5, 5), dtype=uint8)

        grid_A_onehot = zeros((5, 5, 7), dtype=int)
        grid_B_onehot = zeros((5, 5, 7), dtype=int)

        encodings = [a_agent, b_agent, stag, plant, plant] # in order of coords in shipback

        if self._no_plants:
            encodings = encodings[:3]
            shipback = shipback[:3]

        for encoding, (x, y) in zip(encodings, shipback): # encode only the 5 entities present
            if encoding==a_agent:
                if self._playerA_done:
                    # perspective of playerA: A agent is self, B is other
                    self.add_encoding_to_cell(grid_A, x, y, a_agent_done) 
                    grid_A_onehot[x,y][encoding_idx_map[a_agent_done]] = 1

                    # perspective of playerB: B is self, A is other
                    # self always encoded as A agent
                    self.add_encoding_to_cell(grid_B, x, y, b_agent_done)
                    grid_B_onehot[x,y][encoding_idx_map[b_agent_done]] = 1
                else:
                    self.add_encoding_to_cell(grid_A, x, y, a_agent) 
                    grid_A_onehot[x,y][encoding_idx_map[a_agent]] = 1

                    self.add_encoding_to_cell(grid_B, x, y, b_agent) 
                    grid_B_onehot[x,y][encoding_idx_map[b_agent]] = 1

            elif encoding==b_agent:
                if self._playerB_done:
                    self.add_encoding_to_cell(grid_A, x, y, b_agent_done)
                    grid_A_onehot[x,y][encoding_idx_map[b_agent_done]] = 1

                    self.add_encoding_to_cell(grid_B, x, y, a_agent_done)
                    grid_B_onehot[x,y][encoding_idx_map[a_agent_done]] = 1
                else:
                    self.add_encoding_to_cell(grid_A, x, y, b_agent) 
                    grid_A_onehot[x,y][encoding_idx_map[b_agent]] = 1

                    self.add_encoding_to_cell(grid_B, x, y, a_agent)
                    grid_B_onehot[x,y][encoding_idx_map[a_agent]] = 1
            else:
                self.add_encoding_to_cell(grid_A, x, y, encoding)
                grid_A_onehot[x,y][encoding_idx_map[encoding]] = 1

                self.add_encoding_to_cell(grid_B, x, y, encoding)
                grid_B_onehot[x,y][encoding_idx_map[encoding]] = 1

        if self._obs_type=="grid_onehot":
            return grid_A_onehot.flatten(), grid_B_onehot.flatten()

        return grid_A, grid_B

    def add_encoding_to_cell(self, grid, row, col, new_encoding):
        """
        ! FOR GRID OBSERVATION SPACE !
        Add a new encoding to the grid cell at (row, col).
        
        If the cell already contains encodings, append the new encoding
        to the existing binary representation. If the cell is empty (no encoding),
        simply add the new encoding.
        
        Parameters:
        - grid: The numpy grid representing the environment.
        - row: The row index of the cell to modify.
        - col: The column index of the cell to modify.
        - new_encoding: The new encoding to add, given as a bit mask.
        
        Returns:
        None. The modification is done in-place.
        """
        # Check if the cell already contains any encoding
        if grid[row, col] == 0:
            # Cell is empty, add new encoding directly
            grid[row, col] = new_encoding
        else:
            # Cell already contains encoding(s), append new encoding
            grid[row, col] |= new_encoding

    """
    Movement Methods
    """

    def _seek_agent(self, agent_to_seek):
        """
        Moves the stag towards the specified agent
        :param agent_to_seek: agent to pursue
        :return: new position tuple for the stag
        """
        agent = self.A_AGENT
        if agent_to_seek == "b":
            agent = self.B_AGENT

        move = self._seek_entity(self.STAG, agent)

        return self._move_entity(self.STAG, move)

    def _move_stag(self):
        """
        Moves the stag towards the nearest agent.
        :return:
        """
        if self._stag_follows:
            stag, agents = self.STAG, self.AGENTS
            a_dist = hypot(
                int(agents[0][0]) - int(stag[0]), int(agents[0][1]) - int(stag[1])
            )
            b_dist = hypot(
                int(agents[1][0]) - int(stag[0]), int(agents[1][1]) - int(stag[1])
            )

            if a_dist < b_dist:
                agent_to_seek = "a" if not self._playerA_done else "b"
            else:
                agent_to_seek = "b" if not self._playerB_done else "a"

            self.STAG = self._seek_agent(agent_to_seek)
        else:
            self.STAG = self._move_entity(self.STAG, self._random_move(self.STAG))

    def reset_entities(self):
        """
        Reset all entity positions.
        :return:
        """
        self._playerA_done = False
        self._playerB_done = False
        self._stag_done = False
        self._reset_agents()
        if self._stag_random_respawn:
            self.STAG=place_entity_in_unoccupied_cell(self.AGENTS,(self.GRID_W,self.GRID_H))
        else:
            self.STAG = [self.GRID_W // 2, self.GRID_H // 2]
        
        if self._no_plants:
            self.PLANTS = [[9,9],[9,9]]
        else:
            self.PLANTS = spawn_plants(
                    grid_dims=self.GRID_DIMENSIONS,
                    how_many=self._forage_quantity,
                    used_coordinates=self.AGENTS + [self.STAG],
                )

    """
    Properties
    """

    @property
    def STAG(self):
        return self._stag_pos

    @STAG.setter
    def STAG(self, new_pos):
        self._stag_pos[0], self._stag_pos[1] = new_pos[0], new_pos[1]


    @property
    def PLANTS(self):
        return self._plants_pos

    @PLANTS.setter
    def PLANTS(self, new_pos):
        self._plants_pos = new_pos

    @property
    def ENTITY_POSITIONS(self):
        return {
            "a_agent": self.A_AGENT,
            "b_agent": self.B_AGENT,
            "stag": self.STAG,
            "plants": self.PLANTS,
        }
