import numpy as np
from PIL import Image
from typing import Optional
import os
import argparse
import random
import yaml
from argparse import Namespace
import math
from flask import Flask, request, jsonify
import tempfile

import torch

from vla import load_vla
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)


class MemVLAService:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10, 
        use_ddim: bool = True,
        use_bf16: bool = False,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")

        kwargs = vars(args).copy()
        for k in [
            "model_id_or_path", "saved_model_path", "pretrained_checkpoint",
        ]:
            kwargs.pop(k, None)
        if not kwargs.get("hf_token"):
            kwargs["hf_token"] = os.environ.get("HF_TOKEN")

        self.vla = load_vla(
          model_id_or_path=saved_model_path,
          load_for_training=False,
          **kwargs,
        )
        self.vla = self.vla.to("cuda").eval()
        if use_bf16:
            print("Using bfloat16 inference mode (auto-conversion for all modules).")
            self.vla = self.vla.to(torch.bfloat16)
        else:
            print("Using standard float32 inference mode.")
            self.vla = self.vla.to(torch.float32)

        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        self.args = args
        self.experiment_states = {}
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    @staticmethod
    def _capture_rng_state() -> dict:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        }

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    @staticmethod
    def _validate_snapshot_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError("Snapshot name must contain 1 to 128 characters.")
        return name.strip()

    def save_experiment_state(self, name: str) -> dict:
        if self.action_ensemble:
            raise RuntimeError("The experiment API requires action chunking or single-action inference.")
        name = self._validate_snapshot_name(name)
        if name not in self.experiment_states and len(self.experiment_states) >= 32:
            raise RuntimeError("At most 32 experiment snapshots may be retained at once.")
        self.experiment_states[name] = {
            "model": self.vla.snapshot_runtime_state(),
            "rng": self._capture_rng_state(),
        }
        return self.vla.get_runtime_state_summary()

    def restore_experiment_state(self, name: str) -> dict:
        name = self._validate_snapshot_name(name)
        if name not in self.experiment_states:
            raise KeyError(f"Unknown experiment snapshot: {name}")
        state = self.experiment_states[name]
        self.vla.restore_runtime_state(state["model"])
        self._restore_rng_state(state["rng"])
        return self.vla.get_runtime_state_summary()

    def delete_experiment_state(self, name: str) -> dict:
        name = self._validate_snapshot_name(name)
        if name not in self.experiment_states:
            raise KeyError(f"Unknown experiment snapshot: {name}")
        del self.experiment_states[name]
        return self.vla.get_runtime_state_summary()

    def set_memory_retrieval_enabled(self, enabled: bool) -> dict:
        self.vla.set_memory_retrieval_enabled(enabled)
        return self.vla.get_runtime_state_summary()

    def set_experiment_seed(self, seed: int) -> dict:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.vla.get_runtime_state_summary()

    def get_experiment_status(self) -> dict:
        return {
            "snapshots": sorted(self.experiment_states),
            "memory": self.vla.get_runtime_state_summary(),
        }

    def step(
        self,
        image: str,
        task_description: str = None,
        episode_first_frame: str = 'False',
        return_normalized: bool = False,
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Path to the image file
            task_description: Optional[str], task description
            episode_first_frame: 'True' or 'False', whether the current frame is the first frame of an episode
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        assert episode_first_frame in ['True', 'False']

        if episode_first_frame == 'True':
            self.reset()

        image: Image.Image = Image.open(image)

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)

        # save resized image for debugging
        resized_image.save("resized_image.png")
        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image, 
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale, 
            use_ddim=self.use_ddim, 
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
        )

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = unnormed_actions.tolist()
            normalized_action = normalized_actions[0].tolist()
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
                normalized_action = normalized_actions[:self.action_chunking_window].tolist()
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()
            normalized_action = normalized_actions[0].tolist()

        print(f"Instruction: {task_description}")
        # print(f"Model path: {self.args.saved_model_path} at port {self.args.port}")
        if return_normalized:
            return action, normalized_action
        return action


# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images  
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    #assert h < w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)
    
    image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# Here the image is first center cropped and then resized back to its original size 
# because random crop data augmentation was used during finetuning.
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="")
parser.add_argument("--unnorm_key", type=str, default='custom_finetuning')
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=2345)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)
parser.add_argument("--enable_experiment_api", action="store_true")

args = parser.parse_args()

with open(os.path.join(os.path.dirname(os.path.dirname(args.saved_model_path)), "config.yaml"), "r") as f:
    yaml_args = yaml.safe_load(f) or {}

def deep_update(base: dict, updates: dict):
    """Recursively merge two dictionaries, with updates taking precedence but preserving keys from base."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

cli_args = vars(args)
merged_args = deep_update(yaml_args.copy(), cli_args)

args = Namespace(**merged_args)

inferencer = MemVLAService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    args=args,
)


def _experiment_api_disabled_response():
    if args.enable_experiment_api:
        return None
    return jsonify({"error": "Experiment API is disabled. Start deploy.py with --enable_experiment_api."}), 403


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ready",
        "experiment_api_enabled": bool(args.enable_experiment_api),
    })


@app.route('/process_frame', methods=['POST'])
def inference():
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']

    # Check if text is provided
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    query = request.form['text']

    # Check if episode_first_frame is provided
    if 'episode_first_frame' not in request.form:
        return jsonify({'error': 'No episode_first_frame provided'}), 400
    episode_first_frame = request.form['episode_first_frame']

    return_normalized = request.form.get('return_normalized', 'False') == 'True'

    # Construct input query and prepare for inference
    input_query = {
        'task_description': query,
        'episode_first_frame': episode_first_frame,
    }

    # Save image to a temporary file because the existing preprocessing path expects a path.
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image.save(temp_image.name)
        temp_image_path = temp_image.name

    try:
        result = inferencer.step(temp_image_path, return_normalized=return_normalized, **input_query)
    finally:
        os.unlink(temp_image_path)

    if return_normalized:
        answer, normalized_action = result
    else:
        answer = result
    print(answer)

    # Convert action array to string based on different modes
    if inferencer.action_ensemble:
        # For action ensemble mode, directly convert the action list
        action_str = ' '.join([str(x) for x in answer])
    elif inferencer.action_chunking:
        # For action chunking mode, convert the chunked actions
        action_str = ';'.join([' '.join([str(x) for x in chunk]) for chunk in answer])
    else:
        # For single action mode
        action_str = ' '.join([str(x) for x in answer])

    response = {'response': action_str}
    if return_normalized:
        response.update({
            'action': answer,
            'normalized_action': normalized_action,
            'memory': inferencer.vla.get_runtime_state_summary(),
        })
    return jsonify(response)


@app.route('/experiment/control', methods=['POST'])
def experiment_control():
    disabled_response = _experiment_api_disabled_response()
    if disabled_response is not None:
        return disabled_response

    payload = request.get_json(silent=True) or {}
    action = payload.get('action')

    try:
        if action == 'save':
            memory = inferencer.save_experiment_state(payload.get('name'))
        elif action == 'restore':
            memory = inferencer.restore_experiment_state(payload.get('name'))
        elif action == 'delete':
            memory = inferencer.delete_experiment_state(payload.get('name'))
        elif action == 'set_retrieval':
            enabled = payload.get('enabled')
            if not isinstance(enabled, bool):
                raise ValueError("set_retrieval requires a boolean 'enabled' field.")
            memory = inferencer.set_memory_retrieval_enabled(enabled)
        elif action == 'seed':
            seed = payload.get('seed')
            if not isinstance(seed, int):
                raise ValueError("seed requires an integer 'seed' field.")
            memory = inferencer.set_experiment_seed(seed)
        elif action == 'status':
            return jsonify({'response': inferencer.get_experiment_status()})
        else:
            raise ValueError("Unknown action. Use save, restore, delete, set_retrieval, seed, or status.")
    except (KeyError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 409

    return jsonify({'response': memory})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)
