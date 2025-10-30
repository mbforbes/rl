"""

state is a 4-dimensional array of floats:
    0. cart_position: float, x coordinate in (-4.8, 4.8), center = 0
    1. cart_velocity: float, x velocity coordinate in (-inf, inf), still = 0
    2. pole_angle: float of radians, in (-1/2 pi, 1/2 pi), upright = 0
        - 90 degrees ~= 1.57 radians
        - 12 degrees ~= 0.21 radians
    3. pole_angular_velocity: float, in (-inf, inf), still = 0
        (actually not sure the exact units b/c physics. maybe rad/s?)
"""

import numpy as np

import random


def policy_1param(state: np.ndarray) -> int:
    """150 - 160"""
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    if pole_angle < 0:
        return 0
    return 1


def params_1_param() -> int:
    return 1


def policy_weighted(state: np.ndarray) -> int:
    """trying different angle thresholds (results avg over 50 runs):

    0.01 - 157 +/- 28
    0.1  - 142 +/- 30
    0.2  - 117 +/- 24
    0.3  - 108 +/- 26
    0.4  -  95 +/- 20
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    default_action = 0 if pole_angle < 0 else 1
    other_action = 1 if default_action == 0 else 1

    rng = random.random()
    # at thresh, we want to take the default action 100% of the time
    # at 0,      we want to be fully random
    abs_pole_angel = abs(pole_angle)
    # at 0.00,   rng threshold = 0.5
    # at thresh, rng threshold = 1.0 (default if < thresh, other if >)
    angle_threshold = 0.3
    rng_threshold = 0.5 + min(abs_pole_angel / angle_threshold, 1.0) * 0.5
    if rng < rng_threshold:
        return default_action
    else:
        # print("taking other action")
        return other_action


def params_weighted() -> int:
    return 1


def policy_predictive(state: np.ndarray) -> int:
    """
    Combines pole angle with pole angular velocity to anticipate future position.

    Trying different VELOCITY_WEIGHT values:
                                    (50 runs)
    - 0.0:  baseline (~160)         160
    - 0.05: slight lookahead       3962.52 +/- 1257.910048294392
    - 0.08:                        4212.82 +/- 930.0370033498667
    - 0.1:  moderate lookahead     4179.3 +/- 961.8728242340566
    - 0.12:                        3863.02 +/- 1344.6447782221146
    - 0.15: more anticipation      3508.04 +/- 1523.9145508853178
    - 0.2:  aggressive             2641.1 +/- 1731.188554144233
    - 0.3:  very aggressive        2864.28 +/- 1869.6778229416961
    """
    VELOCITY_WEIGHT = 0.07

    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state

    # Predict where pole is heading
    effective_angle = pole_angle + VELOCITY_WEIGHT * pole_angular_velocity

    # Push in direction to counter the effective lean
    if effective_angle < 0:
        return 0  # push left
    else:
        return 1  # push right


def params_predictive() -> int:
    return 1


def policy_predictive_deadband(state: np.ndarray) -> int:
    """
    Same as above but with deadband - center cart if pole is "close enough" to upright.
    """
    VELOCITY_WEIGHT = 0.1
    DEADBAND = 0.01

    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state

    effective_angle = pole_angle + VELOCITY_WEIGHT * pole_angular_velocity

    # If we're close to balanced, can take a slight modification action. we'll use this
    # to center the cart.
    if abs(effective_angle) < DEADBAND:
        # surprisingly, if we're right, we move RIGHT (!). i think this may have the
        # intuition of riding a bicycle, in that if you want to turn left, you have to
        # turn right first, which swings the bicycle to the left, and then you lean left
        # to make the turn.
        choice = 0 if cart_position > 0 else 1  # use cart position
        # print(
        #     f"in deadband, cart position {cart_position} so returning {'left' if choice == 0 else 'right'}"
        # )
        return choice

    if effective_angle < 0:
        return 0
    else:
        return 1


def params_predictive_deadband() -> int:
    return 2


def policy_predictive_deadband_demo(state: np.ndarray) -> int:
    """simplified code to demonstrate 2-param solution"""
    VELOCITY_WEIGHT = 0.1
    DEADBAND = 0.01

    cart_p, _cart_v, pole_angle, pole_ang_v = state
    effective_angle = pole_angle + VELOCITY_WEIGHT * pole_ang_v

    if abs(effective_angle) < DEADBAND:
        return 1 if cart_p > 0 else 0

    return 0 if effective_angle < 0 else 1
