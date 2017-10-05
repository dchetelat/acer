import numpy as np
import brain
import agent
from torch import multiprocessing as mp
from core import *


def run_agent(shared_brain, render=False):
    local_agent = agent.Agent(shared_brain, render)
    for iteration in range(MAX_ITERATIONS):
        if render:
            print("Iteration #{}".format(iteration), end="")
        local_agent.learn(on_policy=True)
        for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
            local_agent.learn(on_policy=False)


if __name__ == "__main__":
    if NUMBER_OF_AGENTS == 1:
        # Don't bother with multiprocessing if only one agent
        run_agent(brain.brain, render=True)
    else:
        processes = [mp.Process(target=run_agent, args=(brain.brain, True))
                     for _ in range(NUMBER_OF_AGENTS - 1)]
        processes.append(mp.Process(target=run_agent, args=(brain.brain, True)))
        for process in processes:
            process.start()

        for process in processes:
            process.join()
