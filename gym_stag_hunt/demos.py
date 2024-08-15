from time import sleep

from gym_stag_hunt.envs.gym.escalation import EscalationEnv
from gym_stag_hunt.envs.gym.harvest import HarvestEnv
from gym_stag_hunt.envs.gym.hunt import HuntEnv
from gym_stag_hunt.envs.gym.simple import SimpleEnv
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND

ENVS = {
    "CLASSIC": SimpleEnv,
    "HUNT": HuntEnv,
    "HARVEST": HarvestEnv,
    "ESCALATION": EscalationEnv,
}


def print_ep(reward, done=None, obs=None):
    print({"reward": reward, "simulation over": done, "obs": obs})


def dir_parse(key):
    d = {LEFT: "LEFT", UP: "UP", DOWN: "DOWN", RIGHT: "RIGHT", STAND: "STAND"}
    return d[key]


def manual_input():
    i = input()
    if i in ["w", "W"]:
        i = UP
    elif i in ["a", "A"]:
        i = LEFT
    elif i in ["s", "S"]:
        i = DOWN
    elif i in ["d", "D"]:
        i = RIGHT
    elif i in ["x", "X"]:
        i = STAND

    return i


ENV = "HUNT"

if __name__ == "__main__":

    end_ep_on_reward=True
    print('hello')

    env = ENVS[ENV](obs_type="image", enable_multiagent=True, timestep_penalty=0, 
                    end_ep_on_reward=end_ep_on_reward, stag_frozen=False, stag_follows=False, no_plants=False, forage_reward=2, 
                    mauling_punishment=-1, stag_reward=25, grid_size=(5,5),
                    agent_random_respawn=False, stag_random_respawn=False, run_away_after_maul=True, forage_quantity=2, done_bits=False)
    obs = env.reset()

    if end_ep_on_reward:
        dones = (False,False)
        while all(dones)==False:
            env.render(mode='human')

            if not dones[0]:
                my_action_a=manual_input()
            else:
                my_action_a=None

            if not dones[1]:
                my_action_b=manual_input()
            else:
                my_action_b=None
            
            actions = {'player_0':my_action_a, 'player_1':my_action_b}
            obs, rewards, dones, info = env.step(actions=actions)
            print_ep(reward=rewards, done = dones)

            sleep(0.4)
            if ENV == "CLASSIC":
                env.render(rewards=rewards)
            else:
                env.render(mode="human")
    else: 
            for i in range(10):
                env.render(mode='human')

                my_action_a=manual_input()
                my_action_b=manual_input()
       
                actions = {'player_0':my_action_a, 'player_1':my_action_b}
                obs, rewards, dones, info = env.step(actions=actions)
                print_ep(reward=rewards)

                sleep(0.4)
                if ENV == "CLASSIC":
                    env.render(rewards=rewards)
                else:
                    env.render(mode="human")
    print_ep(reward=rewards, done=dones)
    env.close()
    quit()
