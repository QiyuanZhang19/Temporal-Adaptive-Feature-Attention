from gym.envs.registration import register


register(id='PendulumAug-v0',
         entry_point='envs.pendulum_noise_aug:PendulumAug', max_episode_steps=200)


