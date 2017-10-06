import brain
import agent
from torch import multiprocessing as mp
from core import *


def run_agent(shared_brain, render=False):
    """
    Run the agent.

    Parameters
    ----------
    shared_brain : brain.Brain
        The shared brain the agents will use and update.
    render : boolean, optional
        Should the agent render its actions in the on-policy phase?
    """
    local_agent = agent.Agent(shared_brain, render)
    for episode in range(MAX_EPISODES):
        if render:
            print("Episode #{}".format(episode), end="")
        local_agent.run_episode()

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
