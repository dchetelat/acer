import numpy as np
import brain
import agent
from torch import multiprocessing as mp
from core import *


def run_agent(brain_actor_critic, render=False):
    local_agent = agent.Agent(brain_actor_critic, render)
    for iteration in range(MAX_ITERATIONS):
        if render:
            print("Iteration #{}".format(iteration), end="")
        local_agent.learn(on_policy=True)
        for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
            local_agent.learn(on_policy=False)


if __name__ == "__main__":
    processes = [mp.Process(target=run_agent, args=(brain.brain_actor_critic, False))
                 for _ in range(NUMBER_OF_AGENTS - 1)]
    processes.append(mp.Process(target=run_agent, args=(brain.brain_actor_critic, True)))
    for process in processes:
        process.start()

    for process in processes:
        process.join()
