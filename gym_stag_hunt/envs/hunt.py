from gym.spaces import Discrete, Box, MultiDiscrete
from numpy import int64

from gym_stag_hunt.envs.abstract_markov_staghunt import AbstractMarkovStagHuntEnv
from gym_stag_hunt.src.games.staghunt_game import StagHunt


class HuntEnv(AbstractMarkovStagHuntEnv):
    def __init__(self,
                 grid_size=(5, 5),
                 screen_size=(600, 600),
                 obs_type='image',
                 load_renderer=False,
                 episodes_per_game=1000,
                 stag_follows=True,
                 run_away_after_maul=False,
                 forage_quantity=2,
                 stag_reward=5,
                 forage_reward=1,
                 mauling_punishment=-5
                 ):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param screen_size: A (W, H) tuple corresponding to the pixel dimensions of the game window
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        :param episodes_per_game: How many timesteps take place before we reset the entity positions.
        :param stag_follows: Should the stag seek out the nearest agent (true) or take a random move (false)
        :param run_away_after_maul: Does the stag stay on the same cell after mauling an agent (true) or respawn (false)
        :param forage_quantity: How many plants will be placed on the board.
        :param stag_reward: How much reinforcement the agents get for catching the stag
        :param forage_reward: How much reinforcement the agents get for harvesting a plant
        :param mauling_punishment: How much reinforcement the agents get for trying to catch a stag alone (MUST be neg.)
        """
        if not (stag_reward > forage_reward >= 0 > mauling_punishment):
            raise AttributeError('The game does not qualify as a Stag Hunt, please change parameters so that '
                                 'stag_reward > forage_reward >= 0 > mauling_punishment')
        if mauling_punishment == forage_reward:
            raise AttributeError('Mauling punishment and forage reward are equal.'
                                 ' Game logic will not function properly.')
        if episodes_per_game <= 0:
            raise AttributeError('Episodes per game is too low. Please provide a positive integer.')
        total_cells = grid_size[0] * grid_size[1]
        if forage_quantity >= total_cells - 3:  # -3 is for the cells occupied by the agents and stag
            raise AttributeError('Forage quantity is too high. The plants will not fit on the grid.')
        if total_cells < 3:
            raise AttributeError('Grid is too small. Please specify a larger grid size.')

        super(HuntEnv, self).__init__(grid_size=grid_size, obs_type=obs_type)

        self.game_title = 'hunt'
        self.stag_reward = stag_reward
        self.forage_reward = forage_reward
        self.mauling_punishment = mauling_punishment
        self.reward_range = (mauling_punishment, stag_reward)

        window_title = "OpenAI Gym - Stag Hunt (%d x %d)" % grid_size  # create game representation
        self.game = StagHunt(window_title=window_title,
                             grid_size=grid_size,
                             screen_size=screen_size,
                             obs_type=obs_type,
                             load_renderer=load_renderer,
                             episodes_per_game=episodes_per_game,
                             stag_reward=stag_reward,
                             stag_follows=stag_follows,
                             run_away_after_maul=run_away_after_maul,
                             forage_quantity=forage_quantity,
                             forage_reward=forage_reward,
                             mauling_punishment=mauling_punishment)

        self.action_space = Discrete(4)  # up, down, left, right on the grid

        if obs_type == 'image':  # Observation is the rgb pixel array
            self.observation_space = Box(0, 255, shape=(screen_size[0], screen_size[1], 3), dtype=int64)
        elif obs_type == 'coords':  # Observation is an xy matrix with booleans signifying entities in the cell
            self.observation_space = MultiDiscrete([3+forage_quantity, 2])  # TODO: FIX