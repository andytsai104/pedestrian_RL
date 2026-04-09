from ..utils.config_loader import load_config
from ..utils.td3_utils import PedestrianRLEnv
from ..utils.eval_utils import TD3PolicyRunner


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


if __name__ == "__main__":
    run_td3_policy()