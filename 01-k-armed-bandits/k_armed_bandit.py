"""Before seeing the answer, fun to try out.

agent       = our poor gambler. we'll give n_arms globally
environment = per-arm rewards, setup of calling T times independently in succession
state       = turn t (int), total turns T (int)
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

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ARMS = [
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

TURNS = 1000
"""each turn will be in [0, turns - 1]"""


def reward_fn(action: int):
    assert action >= 0 and action < len(ARMS)
    return np.random.normal(ARMS[action], 1)


@dataclass
class State:
    turn: int
    turns: int


class Agent(ABC):
    __slots__ = ("history", "n_arms", "total_reward")

    def __init__(self):
        self.history: dict[int, list[float]] = defaultdict(list)
        """action -> rewards"""
        self.n_arms = len(ARMS)
        self.total_reward = 0.0

    def receive_reward(self, state: State, action: int, reward: float):
        """state is passed again in case helpful for agent; same as was called in
        policy()"""
        self.history[action].append(reward)
        self.total_reward += reward

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def policy(self, state: State) -> int:
        pass


class RandomAgent(Agent):
    """Randomly picks arm each time"""

    def policy(self, state: State) -> int:
        return np.random.randint(1, self.n_arms)


class StepGreedyAgent(Agent):
    """Runs randomly until turn t, then exploits best action."""

    __slots__ = ("exploit_t",)

    def __repr__(self) -> str:
        return f"StepGreedyAgent(exploit_t={self.exploit_t})"

    def __init__(self, exploit_t: int):
        super().__init__()
        self.exploit_t = exploit_t

    def best_action(self) -> int:
        # really should cache this
        averages = [0.0] * self.n_arms
        for a, rewards in self.history.items():
            averages[a] = float(np.mean(rewards))
        return int(np.argmax(averages))

    def policy(self, state: State) -> int:
        if state.turn < self.exploit_t:
            return np.random.randint(1, self.n_arms)
        return self.best_action()


class GreedyTweenAgent(Agent):
    """epsilon-greedy tweening"""

    __slots__ = ("start_e", "begin_tween_t", "end_tween_t", "end_tween_e")

    def __repr__(self) -> str:
        return f"GreedyTweenAgent({self.start_e} [({self.begin_tween_t})->({self.end_tween_t})] {self.end_tween_e})"

    def __init__(
        self, start_e: float, begin_tween_t: int, end_tween_t: int, end_tween_e: float
    ):
        super().__init__()
        self.start_e = start_e
        self.begin_tween_t = begin_tween_t
        self.end_tween_t = end_tween_t
        self.end_tween_e = end_tween_e

    def best_action(self) -> int:
        # really should cache this
        averages = [0.0] * self.n_arms
        for a, rewards in self.history.items():
            averages[a] = float(np.mean(rewards))
        return int(np.argmax(averages))

    def policy(self, state: State) -> int:
        if state.turn < self.begin_tween_t:
            e = self.start_e
        elif state.turn >= self.begin_tween_t and state.turn < self.end_tween_t:
            total_tween_turns = self.end_tween_t - self.begin_tween_t
            tween_progress_turns = state.turn - self.begin_tween_t
            tween_progress = tween_progress_turns / total_tween_turns
            e = self.start_e + tween_progress * (self.end_tween_e - self.start_e)
        else:
            # state.turn > self.end_tween_t:
            e = self.end_tween_e

        # random action if epsilon, else best
        if np.random.uniform() < e:
            return np.random.randint(1, self.n_arms)
        else:
            return self.best_action()


def expected_random_reward() -> float:
    """expected value of total reward for a policy that picks random actions"""
    return float(np.mean(ARMS) * TURNS)


def simulated_random_reward() -> float:
    """total reward of simulation of a policy that takes random actions every turn"""
    # not super efficient. at least sampling from each arm at once.
    samples = TURNS // len(ARMS)  # approx
    total = 0.0
    for _ in range(samples):
        total += float(sum(np.random.normal(loc=ARMS, scale=1)))
    return total


def expected_best_reward() -> float:
    """expected value of total reward for a policy that picks the best action each time"""
    return float(np.max(ARMS) * TURNS)


def simulated_best_reward() -> float:
    """total reward of simulation of a policy that picks the best action each time"""
    return float(sum(np.random.normal(loc=np.max(ARMS), scale=1, size=TURNS)))


def multiple_agents() -> None:
    agents_rewards: dict[Agent, float] = {
        a: 0.0
        for a in [
            RandomAgent(),
            StepGreedyAgent(50),
            GreedyTweenAgent(
                start_e=1.0, begin_tween_t=30, end_tween_t=70, end_tween_e=0.0
            ),
        ]
    }
    for t in range(TURNS):
        for agent in agents_rewards.keys():
            state = State(turn=t, turns=TURNS)
            action = agent.policy(state)
            reward = reward_fn(action)
            agent.receive_reward(state, action, reward)
            agents_rewards[agent] += reward
    for agent, total_reward in agents_rewards.items():
        print(f"Reward for {agent}: {total_reward}")
        if hasattr(agent, "best_action"):
            # if callable(getattr(agent, "best_action", None)):
            print(f"- {agent}'s best action estimate:", agent.best_action())  # type: ignore
    print()
    print("References:")
    print("True best action:", np.argmax(ARMS))
    print(f"Expected random reward: {expected_random_reward()}")
    print(f"Simulated random reward: {simulated_random_reward()}")
    print(f"Expected best reward: {expected_best_reward()}")
    print(f"Simulated best reward: {simulated_best_reward()}")


def plot_best_action(
    title: str, experiment_turn_bestaction: list[tuple[int, int, int]]
):
    etb = experiment_turn_bestaction
    data = pd.DataFrame(
        {
            "turn": [d[1] for d in etb],
            "best_action": [d[2] for d in etb],
            "experiment": [d[0] for d in etb],
        }
    )

    # Plot all runs faintly
    sns.lineplot(
        data=data, x="turn", y="best_action", hue="experiment", alpha=0.7, legend=False
    )
    plt.title(title)

    # Overlay average with shaded region
    # sns.lineplot(
    #     data=data,
    #     x="turn",
    #     y="best_action",
    #     estimator="mean",
    #     errorbar="sd",
    #     color="black",
    # )

    plt.show()


def action_settling():
    experiment_turn_bestaction: list[tuple[int, int, int]] = []
    for experiment in range(10):
        agent = StepGreedyAgent(TURNS)
        for t in range(TURNS):
            state = State(turn=t, turns=TURNS)
            action = agent.policy(state)
            reward = reward_fn(action)
            agent.receive_reward(state, action, reward)
            experiment_turn_bestaction.append((experiment, t, agent.best_action()))

    plot_best_action("StepGreedyAgent", experiment_turn_bestaction)


def main() -> None:
    multiple_agents()
    action_settling()


if __name__ == "__main__":
    main()
