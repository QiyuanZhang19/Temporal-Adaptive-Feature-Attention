import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


slice_s = {'HalfCheetah-v2': [[0, 8], [8, 17]],
               'Walker2d-v2': [[0, 8], [8, 17]],
               'Hopper-v2': [[0, 5], [5, 11]],
               'Ant-v2': [[0, 13], [13, 13 + 98]],
               'PendulumAug-v0': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}


def softmax(x):
    y = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)
    return y


def show_attention(ax, env_name, seed, num_checkpoint, flag_label=False):
    data = pickle.load(open(f'./results/{env_name}/seed_{seed}/policy_{num_checkpoint}_attention_{0}.pkl', 'rb'))
    s = data[:, 0, :]
    s = np.exp(s) / np.sum(np.exp(s), axis=-1, keepdims=True)
    l, feature = s.shape[0], s.shape[1]
    feature_index = slice_s[env_name]
    ys = []
    for i in range(len(feature_index)):
        y = np.sum(s[:l, slice(*feature_index[i])], axis=-1)
        ys.append(y)
    s = np.asarray(ys)
    b, z = s.shape
    x = np.arange(0, z)
    for k in (range(b)):
        ax.plot(x, s[k, :], color=sns.color_palette()[k], label=k, alpha=.8)
    if flag_label:
        if 'Pendulum' not in env_name:
            plt.legend(["Position", "Velocity"])
        else:
            plt.legend([r"$\cos{\phi}$", r"$\sin{\phi}$", r"$\dot{\phi}$", r"$\omega_1$", r"$\omega_2$"])


def show_env_attention(env_name, seed, num_checkpoint):
    fig = plt.figure(figsize=(18, 16), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel("Attention weights")
    ax.set_xlabel("Time Step")
    ax.set_title(f'{env_name}')
    show_attention(ax, env_name, seed=seed, num_checkpoint=num_checkpoint, flag_label=True)
    ax.grid(color="black", which="both", linestyle=':')
    plt.savefig(f"{num_checkpoint}.png")
    # plt.show()


