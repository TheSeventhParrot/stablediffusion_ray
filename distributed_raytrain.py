import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import os
from google.cloud import storage

# Read environment variables for your training config
MODEL_NAME = os.environ.get("MODEL_NAME")
INSTANCE_DIR = os.environ.get("INSTANCE_DIR")
CLASS_DIR = os.environ.get("CLASS_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
VAE_PATH = os.environ.get("VAE_PATH")
GCLOUD_BUCKET = os.environ.get("GCLOUD_BUCKET")

# The DreamBooth training arguments you want to pass to the script.
CMD_LINE_ARGS = [
    f"--pretrained_model_name_or_path={MODEL_NAME}",
    f"--pretrained_vae_model_name_or_path={VAE_PATH}",
    f"--instance_data_dir={INSTANCE_DIR}",
    f"--class_data_dir={CLASS_DIR}",
    f"--output_dir={OUTPUT_DIR}",
    "--mixed_precision=fp16",
    "--with_prior_preservation",
    "--prior_loss_weight=1.0",
    "--instance_prompt=a photo of xyz eldritch cosmic horror, lovecraftian, tentacles, otherworldly, impossibly large, cosmic scale, ancient god, unspeakable horror",
    "--class_prompt=a photo of colossal cosmic entity, lovecraftian horror, incomprehensible size, city-sized, tentacles, alien geometry",
    "--resolution=1024",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=2",
    "--gradient_checkpointing",
    "--use_8bit_adam",
    "--learning_rate=5e-6",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--num_class_images=200",
    "--max_train_steps=800",
]

def train_fn_per_worker(config: dict):
    """
    This function runs on the GPU worker nodes only.
    We do local imports of train_dreambooth_lora_sdxl and bitsandbytes-based code
    so the CPU-only head node never attempts these imports.
    """
    worker_rank = ray.train.get_context().get_world_rank()
    print(f"[train_fn_per_worker] Worker rank={worker_rank} started.")

    # Import GPU-dependent code here
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Force local rank=0 to use GPU:0, if needed
        print(f"Worker {worker_rank} sees GPU: {torch.cuda.get_device_name()}")

    # Now import your DreamBooth script that depends on diffusers & bitsandbytes
    from train_dreambooth_lora_sdxl import parse_args, main as dreambooth_main
    # Parse the command-line arguments you passed via the config
    training_args = parse_args(input_args=config["cmd_line_args"])
    
    # Run the training script
    dreambooth_main(training_args)

    worker_rank = ray.train.get_context().get_world_rank()
    if worker_rank == 0:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCLOUD_BUCKET)
        blob = bucket.blob("monster_lora_model/pytorch_lora_weights.safetensors")
        blob.upload_from_filename(f"{OUTPUT_DIR}/pytorch_lora_weights.safetensors")

def main():
    """
    The main function runs on the CPU-only head node, but it does NOT import bitsandbytes
    or train_dreambooth_lora_sdxl. We only parse the top-level environment
    or pass along config to the trainer.
    """
    print("[main] Setting up Ray TorchTrainer with 2 GPU workers...")

    # We store our CMD_LINE_ARGS in the trainer's config, so the worker function can parse them
    trainer = TorchTrainer(
        train_loop_per_worker=train_fn_per_worker,
        train_loop_config={"cmd_line_args": CMD_LINE_ARGS},
        scaling_config=ScalingConfig(
            num_workers=2,   # we'll use 2 GPU workers across 2 GPU nodes
            use_gpu=True,    # each worker requires a GPU
            # optional: resources_per_worker={"CPU": 4, "GPU": 1}
        ),
    )

    print("[main] Starting trainer.fit() now...")
    result = trainer.fit()
    print(f"[main] Trainer finished. Result: {result}")
    if result.error:
        raise RuntimeError(f"Ray Trainer failed: {result.error}")
    print("[main] Training complete!")    


if __name__ == "__main__":
    # Shut down any existing Ray context, then connect to the KubeRay cluster
    ray.shutdown()
    ray.init(address="auto")
    main()
