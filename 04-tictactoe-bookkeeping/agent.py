"""
let G = expected total future discounted reward

policy maps state to actions
- pi(a|s) <-- G
(is policy a useful concept here? do we just iterate over all actions a? we don't know
the next state because the opponent steps too...)

value is a state's estimated G
- v(s) <-- G

discount factor d in [0, 1]

v(s) = r + d*s' + d^2*s'' + ... + d^n*s'(n)

learning rate lr in [0, 1]

say we track the previous state s, we take action a, which brings us reward r and sends
us to state s'.

v(s) = v(s)(1 - lr) + lr*(r + d*v(s'))
       old                new

hmmmm, this feels wrong


---
(looking at book)

we'll use G for the expected discounted future return. G_t only includes future rewards
from turn t+1 onwards:

    G_t = R_{t+1} + d*R_{t+2} + d^2*R_{t+3} + d^3*R_{t+2} + ...
        = R_{t+1} + d*(R_{t+2} + d*R_{t+3} + d^2*R_{t+2} + ...)
        = R_{t+1} + d*G_{t+1}


the value function of a state is defined according to a policy. informally, then formally:

    v(s) = G_t

    v_{pi}(s) = E_{pi}[G_t | s]
              = sum all future discounted rewards from t+1 onwards


then there's the "action-value function q. does it have a better name? it's also G:

    q(s,a) = G_t

let's walk through the bellman equation briefly and consider which parts we'll know:

v(s) - what we're computing

- pi(a|s) - policy. not sure. for each a, we'll end up with 1-9 potential s' depending
            on the opponent's moves. remember "pi (like p) is an ordinary function; the
            | reminds is it defines a probability distribution over a for each s." I
            guess we could: for every a, compute all possible s's, check v(s'), combine
            them somehow (average? worst?), then use this as the value of a, then
            perform a softmax? or just take the best (1 for best a, 0 for rest?). this
            feels weird, like we've already done the whole bellman thing in one step...
            though I guess we haven't because we're assuming v(s') has all the
            time-discounted rewards

- p(s',r|s,a) - environment. given s and a, can we can compute the probability of each
                outcome s' and r? yes! we know the ground truth dynamics. could we also
                have agents estimate this from playing? i think so...

v(s') - recursive definition. this is looking at the next state. how does this
        bottom-out? if we lookup old values in a cache, then ... maybe it works?

ok, so let's do two things:
    1. imagine we have the functions above, write the code to use them to compute v(s)
    2. assume a random policy and write the simplest versions of the functions above

questions / todos:
- i want to render as a heatmap
- is there an *update* rule I'm missing?
- do i need to re-estimate all of these every time i change my policy?
     - v(s)        - should be yes, as it's v_{pi}
     - pi(a|s)     - should be yes by definition, this *is* the policy
     - p(s',r|s,a) - I think *no,* this is (our estimate of) the environment dynamics


---

ok, so i got the bellman equation working "correctly," which is great! it currently runs
infinitely. so even tic tac toe's measly set of states is too many without memoization
(unless I have a bug.)

next things to explore:
- system of equations view
- fixed point equation
- Bellman "operator"
- memoized recursion
- dynamic programming w/
    - value iteration
    - policy evaluation
"""

import code
from typing import Callable, Any

import numpy as np

from tictactoe import TicTacToe, TicTacToeOptions, check_board


State = bytes
"""serialized board"""

Action = tuple[int, int]
"""move to (row, col) (both 0-based)"""

Probability = float

Reward = float
"""individual reward value"""

G = float
"""expected discounted return - i.e., expected sum of all future discounted rewards"""

Discount = float


def serialize(board: np.ndarray):
    return board.tobytes()


def deserialize(serialized: bytes) -> np.ndarray:
    return np.frombuffer(serialized, dtype="uint8").reshape(3, 3).copy()


def possible_actions(s: State) -> list[Action]:
    """Returns the set (list) of all possible (row, col) actions for s."""
    board = deserialize(s)
    return [tuple(x) for x in np.argwhere(board == 0).tolist()]


def possible_next_states(s: State, a: Action) -> list[State]:
    """given s and a, what are all possible s'?"""
    board = deserialize(s)
    row, col = a
    assert board[row][col] == 0
    # code.interact(local=dict(globals(), **locals()))
    board[row][col] = 1
    possible_opponent_actions = possible_actions(serialize(board))
    next_states: list[State] = []
    for o_row, o_col in possible_opponent_actions:
        s_prime = board.copy()
        s_prime[o_row][o_col] = 2
        next_states.append(serialize(s_prime))
    return next_states


def possible_rewards(s: State, a: Action, s_prime: State) -> list[Reward]:
    """given s, a, and s', what are all possible r?"""
    # really, the logic lives in p(s',r|s,a) so we'll return all here
    # could make smarter because really only one is possible
    # note this changes with per-turn rewards?
    return [-1.0, 1.0, 0.0]


def pi_random(a: Action, state: State) -> Probability:
    actions = possible_actions(state)
    assert a in actions
    return 1 / len(actions)


def p_groundtruth(s_prime: State, r: Reward, s: State, a: Action) -> Probability:
    """Because rewards are deterministic and opponent plays randomly, s_prime is
    uniformly random over possible next s_primes. If the reward doesn't match, the
    probability 0."""
    s_primes = possible_next_states(s, a)
    if s_prime not in s_primes:
        print("ERROR: shouldn't be checking invalid s_prime?")
        return 0.0
    result = check_board(deserialize(s_prime))
    true_reward = {
        -1: 0.0,
        0: 0.0,
        1: 1.0,
        2: -1.0,
    }[result]
    if r != true_reward:
        return 0.0
    return 1.0 / len(s_primes)


def v(
    s: State,
    pi: Callable[[Action, State], Probability],
    p: Callable[[State, Reward, State, Action], Probability],
    gamma: Discount,
    v_next: Any,
) -> G:
    """bellman equation for v_{pi}

    NOTE: functions need more context
    - pi: needs access to data to not choose randomly. can pi depend on v? because v
          depends on pi?
    - p_groundtruth: needs rewards passed, or could estimate this.
    """
    total: G = 0.0
    for a in possible_actions(s):
        p_action: Probability = pi(a, s)
        g_action: G = 0.0
        for s_prime in possible_next_states(s, a):
            for r in possible_rewards(s, a, s_prime):
                prob = p(s_prime, r, s, a)
                value = r + gamma * v_next(s_prime, pi, p, gamma, v_next)
                g_action += prob * value
        total += p_action * g_action
    return total

    # sum over all available actions:
    #   - pi(a|s) (probability of policy taking a)
    #   - G(a)
    #
    # to compute G(action), we compute
    # for action a, sum over all next states s':
    #   - sum over all possible rewards (r):
    #       - probability p(s',r|a,s) times
    #       - reward r(a,s) + (discount gamma * v(s'))


def main() -> None:
    opts = TicTacToeOptions(
        render_mode="text",
        reward_win=1.0,
        reward_lose=-1.0,
        reward_turn=0.0,
        reward_draw=0.0,
        opponent="random",
    )
    env = TicTacToe(opts)
    board, _info = env.reset()
    env.render()
    while True:
        row, col = -1, -1
        while row == -1 or col == -1:
            try:
                move = input("move: ")
                row, col = [int(x) for x in move.split()]
            except:
                print("Error processing move")

        # print current value
        value = v(serialize(board), pi_random, p_groundtruth, 0.9, v)
        print("Board's current value", value)
        # TODO: try to display next values

        board, reward, terminated, _truncated, _info = env.step((row, col))
        env.render()
        if terminated:
            print("final reward:", reward)
            break


if __name__ == "__main__":
    main()
