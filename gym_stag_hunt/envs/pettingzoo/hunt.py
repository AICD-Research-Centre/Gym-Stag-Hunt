from gym_stag_hunt.envs.gym.hunt import HuntEnv
from gym_stag_hunt.envs.pettingzoo.shared import PettingZooEnv
from pettingzoo.utils import parallel_to_aec


def env(**kwargs):
    return ZooHuntEnvironment(**kwargs)


def raw_env(**kwargs):
    return parallel_to_aec(env(**kwargs))


class ZooHuntEnvironment(PettingZooEnv):
    metadata = {"render_modes": ["human", "array"], "name": "hunt_pz", "is_parallelizable":True}

    def __init__(
        self,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="image",
        enable_multiagent=False,
        opponent_policy="random",
        load_renderer=False,
        stag_follows=True,
        stag_frozen=False,
        stag_random_respawn=False,
        agent_random_respawn=False,
        run_away_after_maul=False,
        forage_quantity=2,
        stag_reward=5,
        forage_reward=1,
        mauling_punishment=-5,
        timestep_penalty=0,
        end_ep_on_reward=False,
        no_plants=False,
        done_bits=False
    ):
        hunt_env = HuntEnv(
            grid_size,
            screen_size,
            obs_type,
            enable_multiagent,
            opponent_policy,
            load_renderer,
            stag_follows,
            stag_frozen,
            stag_random_respawn,
            agent_random_respawn,
            run_away_after_maul,
            forage_quantity,
            stag_reward,
            forage_reward,
            mauling_punishment,
            timestep_penalty,
            end_ep_on_reward,
            no_plants,
            done_bits
        )
        super().__init__(og_env=hunt_env)
