import os
import sys
import traceback

import cv2
import numpy as np
import torch
import tqdm
import yaml
import triton_python_backend_utils as pb_utils

# from omegaconf import OmegaConf
# from saicinpainting.training.trainers import load_checkpoint  
# from torch.utils.data._utils.collate import default_collate

# from saicinpainting.training.data.datasets import make_default_val_dataset
# from saicinpainting.training.trainers import load_checkpoint
# from saicinpainting.utils import register_debug_signal_handlers
# from saicinpainting.evaluation.utils import move_to_device
# from saicinpainting.evaluation.refinement import refine_predict

# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# lama_root_in_model_repo = os.path.join(current_script_dir, 'lama')
# sys.path.insert(0, current_script_dir) # Add model's own directory to path
# print(f"LAMA Full PythonBackend: sys.path modified, current_dir added: {current_script_dir}", flush=True)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
    from omegaconf import OmegaConf
    from saicinpainting.training.trainers import load_checkpoint  
    print("LAMA Full PythonBackend: Successfully imported OmegaConf and load_checkpoint.", flush=True)
except  ImportError as e:
    print(f"LAMA Full PythonBackend: FATAL ERROR importing LaMa components: {e}", flush=True)
    print(e)
    # import traceback
    # traceback.print_stack(e)
    raise RuntimeError(f"Failed to import LaMa components: {e}") from e

class TritonPythonModel:
    def initialize(self, args):
        print("LAMA Full PythonBackend: initialize() CALLED.", flush=True) #

        self.device_str = f"cuda:{args['model_instance_device_id']}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        print(f"LAMA Full PythonBackend: Initializing on device: {self.device_str}", flush=True)

        
        base_path_for_this_model_version = os.path.join(
            args['model_repository'],
            str(args['model_version'])
        )
        print(f"LAMA Full PythonBackend: Base path for this model version: {base_path_for_this_model_version}", flush=True)

        checkpoint_assets_subdir_name = "big-lama"
        self.checkpoint_assets_path = os.path.join(base_path_for_this_model_version, checkpoint_assets_subdir_name)
        print(f"LAMA Full PythonBackend: Resolved checkpoint assets path: {self.checkpoint_assets_path}", flush=True)

        train_config_path = os.path.join(self.checkpoint_assets_path, 'config.yaml')
        print(f"LAMA Full PythonBackend: Attempting to load train_config from: {train_config_path}", flush=True)
        if not os.path.exists(train_config_path):
            error_msg = f"Training config NOT FOUND: {train_config_path}"
            print(f"LAMA Full PythonBackend: ERROR - {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)
        
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True

        checkpoint_filename_relative = "models/best.ckpt"
        checkpoint_path = os.path.join(self.checkpoint_assets_path, checkpoint_filename_relative)
        print(f"LAMA Full PythonBackend: Attempting to load checkpoint from: {checkpoint_path}", flush=True)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"LAMA PythonBackend: Loading checkpoint '{checkpoint_path}' with train_config...", flush=True)
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()
        print("LAMA PythonBackend: LaMa PyTorch model loaded and moved to device.", flush=True)

        if hasattr(self.model, 'generator'):
            self.generator = self.model.generator
            print("LAMA PythonBackend: Accessed self.generator from loaded model.", flush=True)
        else:
            print("LAMA PythonBackend: WARNING - 'generator' attribute not found on loaded model. Assuming loaded model IS the generator.", flush=True)
            self.generator = self.model

    def execute(self, requests):
        responses = []
        for request in requests:
            image_tensor_np = pb_utils.get_input_tensor_by_name(request, "IMAGE_IN").as_numpy()
            mask_tensor_np = pb_utils.get_input_tensor_by_name(request, "MASK_IN").as_numpy()

            image_torch = torch.from_numpy(image_tensor_np).to(self.device)
            mask_torch = torch.from_numpy(mask_tensor_np).to(self.device)

            print(f"LAMA PythonBackend: Received image_torch shape: {image_torch.shape}", flush=True)
            print(f"LAMA PythonBackend: Received mask_torch shape: {mask_torch.shape}", flush=True)

            image_with_holes = image_torch * (1 - mask_torch)
            input_4_channel_generator = torch.cat([image_with_holes, mask_torch], dim=1)
            print(f"LAMA PythonBackend: Prepared 4-channel input shape: {input_4_channel_generator.shape}", flush=True)

            with torch.no_grad():
                inpainted_image_torch = self.generator(input_4_channel_generator)

            print(f"LAMA PythonBackend: Generator output shape: {inpainted_image_torch.shape}", flush=True)
            
            inpainted_image_np = inpainted_image_torch.cpu().numpy()

            out_tensor = pb_utils.Tensor("INPAINTED_OUT", inpainted_image_np)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
            print("LAMA PythonBackend: Response prepared.", flush=True)
        return responses
    
    def finialize(self):
        print('LAMA PythonBackend: Finalizing LaMa model instance.', flush=True)
        self.model = None
        self.generator = None
        torch.cuda.empty_cache()

