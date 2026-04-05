import os
import cv2
from ..utils.config_loader import load_config
from ..utils.eval_utils import PolicyRunner
from ..models.bc_model import BehaviorCloningPolicy


def main():
    config_name = "training_config.json"
    config = load_config(config_name=config_name)
    checkpoint_name = "best_model.pt"
    checkpoint_seed_dir = os.path.join("seed_2")
    checkpoint_path = os.path.join(config["bc"]["checkpoint_dir"], checkpoint_seed_dir, checkpoint_name)

    model_name = "BC"
    runner = PolicyRunner(
        model_class=BehaviorCloningPolicy,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )

    try:
        runner.run(render_bev=True)
    except KeyboardInterrupt:
        print(f"\n[{model_name} PolicyRunner] Stopped by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
