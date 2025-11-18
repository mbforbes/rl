# TODO: make RL (gym) env

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class TicTacToeOptions:
    render_mode: Literal["none", "text"]
    reward_win: float
    reward_lose: float
    reward_turn: float
    reward_draw: float
    opponent: Literal["random"]


def check_board(board: np.ndarray):
    """Returns -1 (draw), 0 (continue), 1 (player 1 won), or 2 (player 2 won)."""
    p1 = board == 1
    p2 = board == 2
    for p_cells, p_num in [(p1, 1), (p2, 2)]:
        col_win = any(p_cells.sum(axis=0) == 3)
        row_win = any(p_cells.sum(axis=1) == 3)
        d1_win = int(np.diag(p_cells).sum()) == 3
        d2_win = int(np.diag(np.fliplr(p_cells)).sum()) == 3
        if col_win or row_win or d1_win or d2_win:
            return p_num
    if int((board == 0).sum()) == 9:
        return -1
    return 0


def random_opponent(board: np.ndarray):
    """Returns 0-based (row, col) to play"""
    n_options = int((board == 0).sum())
    assert n_options > 0 and n_options <= 9
    choice = np.random.choice(n_options)  # 0-based
    seen = -1
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                seen += 1
                if seen == choice:
                    return row, col
    raise ValueError(f"Should have played a move on board: {board}")


class TicTacToe:
    __slots__ = ["options", "board", "terminated", "reward"]

    display_map = {0: " ", 1: "X", 2: "O"}

    def __init__(self, options: TicTacToeOptions):
        """
        Player  | Rendered as | Internal representation
        (empty) |   ' '       | 0
        1       |   'X'       | 1
        2       |   'O'       | 2
        """
        self.options = options
        # NOTE: user must call reset()!

    def step(self, action: tuple[int, int]):
        """Play one move.

            Arguments:
            - action: 0-based (row, col), each in 0..2

            Returns 5-tuple:
            - observation (np.ndarray; the board)
            - reward (float, based on reward policy)
            - terminated (bool, whether the game is over)
            - truncated (bool, always False)
            - info (result of _info())

        if terminated or truncated:
            observation, info = env.reset()
        """
        if self.terminated:
            raise ValueError(f"Can't step on terminated board: {self.render()}")
        row, col = action
        if self.board[row][col] != 0:
            raise ValueError(f"Illegal move {action} on board: {self.render()}")

        # play
        self.board[row][col] = 1
        self.reward += self.options.reward_turn

        # determine fate
        result = self._check_and_update()
        if result == 0:
            # nonterminal state: opponent (player 2) plays. then check again.
            op_row, op_col = random_opponent(self.board)
            self.board[op_row][op_col] = 2
            result = self._check_and_update()

        # observation, reward, terminated, truncated, info
        return self.board.copy(), self.reward, self.terminated, False, self._info()

    def reset(self):
        """Resets game to initial state, returns (obs, info)"""
        self.board = np.zeros((3, 3), dtype="uint8")
        self.terminated = False
        self.reward = 0.0
        return self.board.copy(), self._info()

    def render(self):
        if self.options.render_mode == "none":
            return
        elif self.options.render_mode == "text":
            print()
            print(
                "\n---------\n".join(
                    [
                        " | ".join(TicTacToe.display_map[cell] for cell in row)
                        for row in self.board
                    ]
                )
            )
            print()
            return
        else:
            raise ValueError(f"Unsupported render mode: {self.options.render_mode}")

    def _info(self):
        return None

    def _check_and_update(self):
        """Runs check, updates self.{terminated, reward}, and returns result of check."""
        result = check_board(self.board)
        if result != 0:
            # terminal state
            self.terminated = True
            if result == -1:
                self.reward += self.options.reward_draw
            elif result == 1:
                self.reward += self.options.reward_win
            elif result == 2:
                self.reward += self.options.reward_lose
            else:
                raise ValueError(f"Unknown nonzero result {result}")
        return result


def main() -> None:
    # just for testing out the class
    opts = TicTacToeOptions(
        render_mode="text",
        reward_win=1.0,
        reward_lose=-1.0,
        reward_turn=0.0,
        reward_draw=0.0,
        opponent="random",
    )
    env = TicTacToe(opts)
    _board, _info = env.reset()
    env.render()
    while True:
        row, col = -1, -1
        while row == -1 or col == -1:
            try:
                move = input("move: ")
                row, col = [int(x) for x in move.split()]
            except:
                print("Error processing move")
        obs, reward, terminated, _truncated, _info = env.step((row, col))
        env.render()
        if terminated:
            print("final reward:", reward)
            break


if __name__ == "__main__":
    main()
