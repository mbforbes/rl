"""Before seeing the answer, fun to try out.

agent       = our poor gambler
environment = per-arm rewards, setup of calling T times independently in succession
state       = turn t (int), total turns T (int), also we'll give n_arms (int) to avoid sending
action      = k (int, len(arms)) choices
value fn    = since rewards are immediate and drawn from unchanging distributions directly
              keyed from actions, is this relevant? (TODO)
policy      = s -> a. trying a few different ones:
                - ε-greedy, start with ε=1.0 (full random), then turning down to 0 at
                  some time t (maybe at X%? or after X samples?)
                - ε-greedy, gradual descent from x -> y
                - ε-greedy, fixed
reward      = (s, a) --> r. ignores s. payout from action a is bandit arm
              𝒩(μ=arms[a], σ²=1)
model       = (s, a) --> (s', r). we know that t++, T fixed, which is the complete
              state. we want to estimate r.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

arms = [
    1.12894754,
    -1.28650587,
    2.31310725,
    -2.40969583,
    1.41934198,
    2.33873079,
    -1.70843995,
    1.32468109,
    2.46107037,
    -0.19062953,
]
"""true (secret) payout means. payouts for bandit i drawn from unit normal centered at
arms[i], i.e., 𝒩(μ=arms[i], σ²=1). keeping static for easier comparison across runs."""

turns = 1000
"""each turn will be in [0, turns - 1]"""


def reward_fn(action: int):
    assert action >= 0 and action < len(arms)
    return np.random.normal(arms[action], 1)


@dataclass
class State:
    turn: int
    turns: int
    n_arms: int
    """to not let them access `arms` directly (silly)"""


class RandomAgent:
    __slots__ = (
        "history",
        "total_reward",
    )

    def __repr__(self) -> str:
        return "RandomAgent"

    def __init__(self):
        self.history: dict[int, list[float]] = defaultdict(list)
        """action -> rewards. not using. pull into superclass"""
        self.total_reward = 0.0

    def policy(self, state: State) -> int:
        return np.random.randint(1, state.n_arms)

    def get_reward(self, action: int, reward: float):
        """TODO This seems wrong API because (a) should get state? (b) shouldn't have to
        track reward itself to be trusted with it."""
        self.history[action].append(reward)
        self.total_reward += reward


class StepGreedyAgent:
    """Runs randomly until turn t, then exploits best action."""

    __slots__ = (
        "history",
        "total_reward",
        "exploit_t",
    )

    def __repr__(self) -> str:
        return f"StepGreedyAgent(exploit_t={self.exploit_t})"

    def __init__(self, exploit_t: int):
        self.history: dict[int, list[float]] = defaultdict(list)
        """action -> rewards"""
        self.exploit_t = exploit_t
        self.total_reward = 0.0

    def policy(self, state: State) -> int:
        if state.turn < self.exploit_t:
            return np.random.randint(1, state.n_arms)

        # really should cache this
        averages = [0.0] * state.n_arms
        for a, rewards in self.history.items():
            averages[a] = float(np.mean(rewards))
        return int(np.argmax(averages))

    def get_reward(self, action: int, reward: float):
        # hmm, maybe want state. how is this usually API'd?
        self.history[action].append(reward)
        self.total_reward += reward


def expected_random_reward() -> float:
    return float(np.mean(arms) * turns)


def simulated_random_reward() -> float:
    """example simulation of taking random actions every turn for this many turns"""
    # not super efficient. at least sampling from each arm at once.
    samples = turns // len(arms)  # approx
    total = 0.0
    for _ in range(samples):
        total += float(sum(np.random.normal(loc=arms, scale=1)))
    return total


def expected_best_reward() -> float:
    return float(np.max(arms) * turns)


def simulated_best_reward() -> float:
    return float(sum(np.random.normal(loc=np.max(arms), scale=1, size=turns)))


def main() -> None:
    agents = [
        RandomAgent(),
        # TODO: try more and plot
        StepGreedyAgent(10),  # wow very few still works pretty well
    ]
    for t in range(turns):
        for agent in agents:
            action = agent.policy(state=State(t, turns, len(arms)))
            reward = reward_fn(action)
            agent.get_reward(action, reward)
    for agent in agents:
        print(f"Reward for {agent}: {agent.total_reward}")
        # TODO: show best action
    print()
    print("References:")
    print(f"Expected random reward: {expected_random_reward()}")
    print(f"Simulated random reward: {simulated_random_reward()}")
    print(f"Expected best reward: {expected_best_reward()}")
    print(f"Simulated best reward: {simulated_best_reward()}")


if __name__ == "__main__":
    main()
