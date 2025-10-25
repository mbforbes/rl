import code

import gymnasium as gym
import pygame

import cart_pole_env_custom  # really bad, but this has the side effect of registering it!

env = gym.make("CartPoleCustom-v0", render_mode="human", threshold_degrees=90)
# env = gym.make("MountainCar-v0", render_mode="human")
clock = pygame.time.Clock()

n_episodes = 15
best_reward = 0
seed = None
for episode in range(n_episodes):
    observation, info = env.reset(seed=(seed if episode == 0 else None))
    # print(f"Starting observation: {observation}")
    # print(f"Starting info: {info}")

    episode_over = False
    total_reward: float = 0.0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        # action = (
        #     env.action_space.sample()
        # )

        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            # print("Right held")
            action = 1

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward  # type: ignore
        episode_over = terminated or truncated

        # TODO: better to limit in the simulation itself?
        # clock.tick(20)

    best_reward = max(best_reward, total_reward)
    print(f"Episode finished! Total reward: {total_reward}")

# code.interact(local=dict(globals(), **locals()))
env.close()
print(f"Run finished! Best reward: {best_reward}")
