import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..utils.td3_utils import PedestrianRLEnv
from ..models.td3_model import TD3Agent
from ..utils.config_loader import load_config


# ----- plotting helpers -----
def save_json(data, save_path):
    '''Save one dictionary or list to json.''' 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {save_path}")


def smooth_curve(values, window=1):
    '''Smooth one curve with moving average without shrinking the boundary.''' 
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0 or window <= 1:
        return values

    window = int(max(1, round(window)))
    window = min(window, len(values))

    if window % 2 == 0:
        window += 1
        window = min(window, len(values))
        if window % 2 == 0:
            window -= 1

    if window <= 1:
        return values

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode='valid')


def set_plot_style():
    '''Set a clean plotting style for TD3 learning curves.''' 
    plt.rcParams.update({
        'figure.dpi': 130,
        'savefig.dpi': 400,
        'ps.fonttype': 42,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'axes.facecolor': '#f4f4f4',
        'figure.facecolor': 'white',
        'grid.color': '#9a9a9a',
        'grid.alpha': 0.35,
        'grid.linewidth': 0.7,
        'lines.linewidth': 2.2,
        'lines.solid_capstyle': 'round',
    })


def save_figure(fig, save_path):
    '''Save one figure as png.''' 
    save_root, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_root), exist_ok=True)
    fig.savefig(save_root + '.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_root}.png")


def plot_single_curve(values,
                      title,
                      ylabel,
                      save_path,
                      smooth_window=1,
                      show_raw=True,
                      ylim=None):
    '''Plot one TD3 learning curve.''' 
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    x = np.arange(1, len(values) + 1, dtype=np.float32)
    smooth_values = smooth_curve(values, smooth_window)

    if show_raw and smooth_window > 1 and len(values) > 1:
        ax.plot(x, values, alpha=0.18, linewidth=1.0)
    ax.plot(x, smooth_values, label='Smoothed')

    ax.set_title(title, pad=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=False)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_two_curves(values_a,
                    values_b,
                    label_a,
                    label_b,
                    title,
                    ylabel,
                    save_path,
                    smooth_window=1,
                    ylim=None):
    '''Plot two TD3 curves in one figure.''' 
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)

    if len(values_a) == 0 and len(values_b) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    if len(values_a) > 0:
        x_a = np.arange(1, len(values_a) + 1, dtype=np.float32)
        ax.plot(x_a, smooth_curve(values_a, smooth_window), label=label_a)

    if len(values_b) > 0:
        x_b = np.arange(1, len(values_b) + 1, dtype=np.float32)
        ax.plot(x_b, smooth_curve(values_b, smooth_window), label=label_b)

    ax.set_title(title, pad=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=False)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_rate_curves(success_values,
                     collision_values,
                     save_path,
                     smooth_window=10):
    '''Plot rolling success and collision rates.''' 
    success_values = np.asarray(success_values, dtype=np.float32)
    collision_values = np.asarray(collision_values, dtype=np.float32)

    if len(success_values) == 0 and len(collision_values) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    if len(success_values) > 0:
        x_success = np.arange(1, len(success_values) + 1, dtype=np.float32)
        ax.plot(x_success, smooth_curve(success_values, smooth_window), label='Goal reached rate')

    if len(collision_values) > 0:
        x_collision = np.arange(1, len(collision_values) + 1, dtype=np.float32)
        ax.plot(x_collision, smooth_curve(collision_values, smooth_window), label='Collision rate')

    ax.set_title('Episode outcome rate', pad=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=False)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_reward_terms(history, save_path, smooth_window=10):
    '''Plot episode reward-term breakdown.''' 
    if len(history) == 0:
        return

    term_names = [
        'collision',
        'approach_vehicle',
        'goal_progress',
        'stall',
        'living',
        'lane',
        'goal_reached',
    ]

    valid_terms = []
    for term_name in term_names:
        values = [episode.get('reward_terms', {}).get(term_name, 0.0) for episode in history]
        if np.any(np.abs(np.asarray(values, dtype=np.float32)) > 1e-8):
            valid_terms.append((term_name, values))

    if len(valid_terms) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for term_name, values in valid_terms:
        x = np.arange(1, len(values) + 1, dtype=np.float32)
        ax.plot(x, smooth_curve(values, smooth_window), label=term_name)

    ax.set_title('Episode reward-term breakdown', pad=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward contribution')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=False, ncol=2)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_td3_training_results(history,
                              save_dir,
                              episode_smooth_window=10,
                              loss_smooth_window=10,
                              rate_smooth_window=10):
    '''Save all TD3 training plots from one history list.''' 
    os.makedirs(save_dir, exist_ok=True)

    if len(history) == 0:
        return

    rewards = [episode.get('reward', 0.0) for episode in history]
    steps = [episode.get('steps', 0) for episode in history]
    critic_losses = [episode.get('critic_loss', np.nan) for episode in history]
    actor_losses = [episode.get('actor_loss', np.nan) for episode in history]
    final_goal_distance = [episode.get('final_goal_distance', np.nan) for episode in history]
    min_vehicle_distance = [episode.get('min_vehicle_distance', np.nan) for episode in history]
    drivable_ratio = [episode.get('drivable_ratio', np.nan) for episode in history]
    stall_ratio = [episode.get('stall_ratio', np.nan) for episode in history]
    goal_reached = [float(bool(episode.get('goal_reached', False))) for episode in history]
    collision = [float(bool(episode.get('collision', False))) for episode in history]

    # remove NaN-only curves
    critic_losses = np.asarray(critic_losses, dtype=np.float32)
    actor_losses = np.asarray(actor_losses, dtype=np.float32)
    final_goal_distance = np.asarray(final_goal_distance, dtype=np.float32)
    min_vehicle_distance = np.asarray(min_vehicle_distance, dtype=np.float32)
    drivable_ratio = np.asarray(drivable_ratio, dtype=np.float32)
    stall_ratio = np.asarray(stall_ratio, dtype=np.float32)

    plot_single_curve(
        values=rewards,
        title='Episode reward',
        ylabel='Reward',
        save_path=os.path.join(save_dir, 'episode_reward.png'),
        smooth_window=episode_smooth_window,
        show_raw=True,
    )

    plot_single_curve(
        values=steps,
        title='Episode length',
        ylabel='Steps',
        save_path=os.path.join(save_dir, 'episode_length.png'),
        smooth_window=episode_smooth_window,
        show_raw=True,
    )

    if np.isfinite(critic_losses).any():
        plot_single_curve(
            values=np.nan_to_num(critic_losses, nan=np.nanmedian(critic_losses[np.isfinite(critic_losses)]) if np.isfinite(critic_losses).any() else 0.0),
            title='Critic loss',
            ylabel='Loss',
            save_path=os.path.join(save_dir, 'critic_loss.png'),
            smooth_window=loss_smooth_window,
            show_raw=True,
        )

    valid_actor = actor_losses[np.isfinite(actor_losses)]
    if valid_actor.size > 0:
        filled_actor_losses = np.where(
            np.isfinite(actor_losses),
            actor_losses,
            np.nanmedian(valid_actor),
        )
        plot_single_curve(
            values=filled_actor_losses,
            title='Actor loss',
            ylabel='Loss',
            save_path=os.path.join(save_dir, 'actor_loss.png'),
            smooth_window=loss_smooth_window,
            show_raw=True,
        )

    if np.isfinite(final_goal_distance).any():
        plot_single_curve(
            values=np.nan_to_num(final_goal_distance, nan=np.nanmedian(final_goal_distance[np.isfinite(final_goal_distance)])),
            title='Final goal distance',
            ylabel='Distance (m)',
            save_path=os.path.join(save_dir, 'final_goal_distance.png'),
            smooth_window=episode_smooth_window,
            show_raw=True,
        )

    if np.isfinite(min_vehicle_distance).any():
        plot_single_curve(
            values=np.nan_to_num(min_vehicle_distance, nan=np.nanmedian(min_vehicle_distance[np.isfinite(min_vehicle_distance)])),
            title='Minimum vehicle distance',
            ylabel='Distance (m)',
            save_path=os.path.join(save_dir, 'min_vehicle_distance.png'),
            smooth_window=episode_smooth_window,
            show_raw=True,
        )

    plot_two_curves(
        values_a=drivable_ratio,
        values_b=stall_ratio,
        label_a='Drivable ratio',
        label_b='Stall ratio',
        title='Episode behavior ratio',
        ylabel='Ratio',
        save_path=os.path.join(save_dir, 'behavior_ratio.png'),
        smooth_window=episode_smooth_window,
        ylim=(0.0, 1.0),
    )

    plot_rate_curves(
        success_values=goal_reached,
        collision_values=collision,
        save_path=os.path.join(save_dir, 'outcome_rate.png'),
        smooth_window=rate_smooth_window,
    )

    plot_reward_terms(
        history=history,
        save_path=os.path.join(save_dir, 'reward_terms.png'),
        smooth_window=episode_smooth_window,
    )


class TD3Trainer:
    '''Train TD3 online in CARLA.''' 

    def __init__(self, env, agent, training_config):
        self.env = env
        self.agent = agent
        self.training_config = training_config

        td3_cfg = training_config['td3']
        params = td3_cfg['params']

        self.checkpoint_dir = td3_cfg['checkpoint_dir']
        self.media_dir = td3_cfg['media_dir']

        self.num_episodes = int(params['num_episodes'])
        self.batch_size = int(params['batch_size'])
        self.start_steps = int(params['start_steps'])
        self.updates_per_step = int(params['updates_per_step'])
        self.save_every = int(params['save_every'])
        self.episode_smooth_window = int(params.get('episode_smooth_window', 10))
        self.loss_smooth_window = int(params.get('loss_smooth_window', 10))
        self.rate_smooth_window = int(params.get('rate_smooth_window', 10))

        self.total_env_steps = 0
        self.history = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)

    def save_history(self, save_path):
        '''Save TD3 training history to json.''' 
        save_json(self.history, save_path)

    def save_outputs(self):
        '''Save history json and TD3 learning plots.''' 
        history_path = os.path.join(self.checkpoint_dir, 'td3_training_history.json')
        self.save_history(history_path)

        plot_td3_training_results(
            history=self.history,
            save_dir=self.media_dir,
            episode_smooth_window=self.episode_smooth_window,
            loss_smooth_window=self.loss_smooth_window,
            rate_smooth_window=self.rate_smooth_window,
        )

    def train(self):
        '''Run TD3 training loop.''' 
        reward_term_names = [
            'collision',
            'approach_vehicle',
            'goal_progress',
            'stall',
            'living',
            'lane',
            'goal_reached',
        ]

        for episode_idx in range(1, self.num_episodes + 1):
            obs, reset_info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False
            last_info = reset_info
            last_update_info = None

            collision_flag = False
            drivable_steps = 0
            stall_steps = 0
            final_goal_distance = None
            episode_min_vehicle_distance = float('inf')
            reward_terms_sum = {term_name: 0.0 for term_name in reward_term_names}

            while not (terminated or truncated):
                if self.total_env_steps < self.start_steps:
                    action = self.env.sample_random_action()
                else:
                    action = self.agent.select_action(obs, add_noise=True)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)

                self.agent.store_transition(obs, action, reward, next_obs, done)

                if len(self.agent.replay_buffer) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        update_info = self.agent.train_step(batch_size=self.batch_size)
                        if update_info is not None:
                            last_update_info = update_info

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                self.total_env_steps += 1
                last_info = info

                if bool(info.get('collision', False)):
                    collision_flag = True

                if bool(info.get('on_driving_lane', False)):
                    drivable_steps += 1

                reward_terms = info.get('reward_terms', {})
                for term_name in reward_term_names:
                    reward_terms_sum[term_name] += float(reward_terms.get(term_name, 0.0))

                if float(reward_terms.get('stall', 0.0)) != 0.0:
                    stall_steps += 1

                if info.get('goal_distance', None) is not None:
                    final_goal_distance = float(info['goal_distance'])

                if info.get('min_vehicle_distance', None) is not None:
                    episode_min_vehicle_distance = min(
                        episode_min_vehicle_distance,
                        float(info['min_vehicle_distance'])
                    )

            episode_result = {
                'episode': episode_idx,
                'reward': float(episode_reward),
                'steps': int(episode_steps),
                'termination': last_info.get('term_reason', None),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'goal_reached': bool(last_info.get('term_reason', None) == 'goal_reached'),
                'collision': bool(collision_flag),
                'final_goal_distance': final_goal_distance,
                'min_vehicle_distance': None if episode_min_vehicle_distance == float('inf') else float(episode_min_vehicle_distance),
                'drivable_ratio': float(drivable_steps / max(episode_steps, 1)),
                'stall_ratio': float(stall_steps / max(episode_steps, 1)),
                'buffer_size': len(self.agent.replay_buffer),
                'total_env_steps': self.total_env_steps,
                'reward_terms': reward_terms_sum,
            }

            if last_update_info is not None:
                episode_result['critic_loss'] = last_update_info['critic_loss']
                episode_result['actor_loss'] = last_update_info['actor_loss']
                episode_result['total_updates'] = last_update_info['total_updates']

            self.history.append(episode_result)

            print(
                f"[TD3 Train] Episode [{episode_idx}/{self.num_episodes}] "
                f"reward={episode_reward:.4f}, "
                f"steps={episode_steps}, "
                f"reason={last_info.get('term_reason', None)}, "
                f"collision={collision_flag}, "
                f"goal_reached={episode_result['goal_reached']}, "
                f"buffer={len(self.agent.replay_buffer)}"
            )

            if episode_idx % self.save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'td3_episode_{episode_idx:03d}.pt')
                self.agent.save(checkpoint_path)
                print(f"Saved: {checkpoint_path}")
                self.save_outputs()

        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'td3_last.pt')
        self.agent.save(final_checkpoint_path)
        print(f"Saved: {final_checkpoint_path}")

        self.save_outputs()


class TD3PolicyRunner:
    '''Run one trained TD3 policy in CARLA.''' 

    def __init__(self, env, checkpoint_path, training_config):
        self.env = env
        self.training_config = training_config
        self.agent = build_td3_agent(training_config=training_config, max_speed=env.max_ped_speed, device=env.device)
        self.agent.load(checkpoint_path=checkpoint_path, load_optimizers=False)
        print(f"[TD3PolicyRunner] Loaded checkpoint: {checkpoint_path}")

    def run(self):
        '''Run trained TD3 policy without exploration noise.''' 
        obs, _ = self.env.reset()

        while True:
            action = self.agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = self.env.step(action)

            print(
                f"[TD3 Run] step={info['episode_step']} "
                f"reward={reward:.4f} "
                f"goal_distance={info['goal_distance']} "
                f"min_vehicle_distance={info['min_vehicle_distance']}"
            )

            if terminated or truncated:
                print(f"[TD3 Run] Episode ended: {info['term_reason']}")
                obs, _ = self.env.reset()



def build_td3_agent(training_config, max_speed, device='cuda'):
    '''Build TD3 agent from training_config.json.''' 
    cnn_cfg = training_config['cnn']
    td3_params = training_config['td3']['params']

    agent = TD3Agent(
        input_channels=5,
        bev_feature_dim=cnn_cfg['bev_feature_dim'],
        scalar_feature_dim=7,
        hidden_dim=cnn_cfg['hidden_dim'],
        action_dim=3,
        max_speed=max_speed,
        actor_learning_rate=td3_params['actor_learning_rate'],
        critic_learning_rate=td3_params['critic_learning_rate'],
        actor_weight_decay=td3_params['actor_weight_decay'],
        critic_weight_decay=td3_params['critic_weight_decay'],
        gamma=td3_params['gamma'],
        tau=td3_params['tau'],
        policy_noise=td3_params['policy_noise'],
        noise_clip=td3_params['noise_clip'],
        policy_delay=td3_params['policy_delay'],
        exploration_speed_noise=td3_params['exploration_speed_noise'],
        exploration_direction_noise=td3_params['exploration_direction_noise'],
        replay_capacity=td3_params['replay_capacity'],
        dropout=td3_params['dropout'],
        device=device,
    )

    return agent



def train_td3():
    '''Train TD3 pedestrian policy.''' 
    training_config = load_config('training_config.json')

    env = PedestrianRLEnv(
        sim_config_name='sim_config.json',
        training_config_name='training_config.json',
        no_rendering_mode=True,
        render_bev=False,
        device='cuda',
    )
    agent = build_td3_agent(
        training_config=training_config,
        max_speed=env.max_ped_speed,
        device=env.device,
    )
    trainer = TD3Trainer(
        env=env,
        agent=agent,
        training_config=training_config,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print('\n[TD3 Train] Stopped by user.')
    finally:
        env.close()



def run_td3_policy(checkpoint_path):
    '''Run trained TD3 policy in CARLA.''' 
    training_config = load_config('training_config.json')

    env = PedestrianRLEnv(
        sim_config_name='sim_config.json',
        training_config_name='training_config.json',
        no_rendering_mode=False,
        render_bev=True,
        device='cuda',
    )
    runner = TD3PolicyRunner(
        env=env,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print('\n[TD3 Run] Stopped by user.')
    finally:
        env.close()


if __name__ == '__main__':
    train_td3()
