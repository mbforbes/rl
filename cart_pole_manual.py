import code

import gymnasium as gym
import pygame

# TODO: fork env to add my own and
#  - pass custom limits
#  - render boundaries
#  - render score
env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("MountainCar-v0", render_mode="human")
clock = pygame.time.Clock()

n_episodes = 10
best_reward = 0
for episode in range(n_episodes):
    observation, info = env.reset()  # starts a new episode
    print(f"Starting observation: {observation}")
    print(f"Starting observation: {info}")

    episode_over = False
    total_reward: float = 0.0

    while not episode_over:
        env.render()

        # code.interact(local=dict(globals(), **locals()))

        # Choose an action: 0 = push cart left, 1 = push cart right
        # action = (
        #     env.action_space.sample()
        # )  # Random action for now - real agents will be smarter!

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
        clock.tick(60)

    best_reward = max(best_reward, total_reward)
    print(f"Episode finished! Total reward: {total_reward}")
env.close()
print(f"Run finished! Best reward: {best_reward}")
