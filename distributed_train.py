#!/usr/bin/env python3
import os
import subprocess
import ray

MODEL_NAME = os.environ.get("MODEL_NAME")
INSTANCE_DIR = os.environ.get("INSTANCE_DIR")
CLASS_DIR = os.environ.get("CLASS_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
VAE_PATH = os.environ.get("VAE_PATH")
HF_TOKEN = os.environ.get("HF_TOKEN")
CONFIG_YAML = os.environ.get("CONFIG_YAML")
DB_SCRIPT = os.environ.get("DB_SCRIPT")

def main():
    # 1) Connect to the existing Ray cluster (the head pod in KubeRay).
    ray.init(address="auto")
    print("Ray cluster resources:", ray.cluster_resources())

    # 2) Identify the ALIVE nodes that have GPU resources.
    #    Each node's 'Resources' dict might have "GPU": <count of GPUs>.
    all_nodes = ray.nodes()
    gpu_nodes = [
        node for node in all_nodes
        if node["Alive"] and node["Resources"].get("GPU", 0) >= 1
    ]
    if not gpu_nodes:
        raise RuntimeError("No nodes with GPU resources found in this Ray cluster.")

    # Sort nodes for consistent rank assignment (e.g., by IP).
    gpu_nodes.sort(key=lambda n: n["NodeManagerAddress"])

    # 3) Calculate total number of GPUs across these nodes (for WORLD_SIZE).
    #    Also gather the local GPU counts so we can do per-node spawn.
    local_gpu_counts = []
    total_gpus = 0
    for node in gpu_nodes:
        count = int(node["Resources"]["GPU"])
        local_gpu_counts.append(count)
        total_gpus += count

    print(f"Discovered {len(gpu_nodes)} GPU nodes, for a total of {total_gpus} GPUs.")

    # 4) We'll designate the first node in gpu_nodes as the 'master' (rank 0).
    master_node = gpu_nodes[0]
    master_ip = master_node["NodeManagerAddress"]
    master_port = os.environ.get("MASTER_PORT", "29500")  # or pick any free port

    print(f"Master node IP: {master_ip}, port: {master_port}")

    # 5) Define a Ray remote function that runs on each node.
    #    We'll request all GPUs on that node so we have exclusive access there.
    @ray.remote
    def run_accelerate_on_node(node_rank: int,
                               master_addr: str,
                               master_port: str,
                               world_size: int,
                               gpus_on_this_node: int):
        """
        A remote task that sets up multi-node environment variables,
        then calls accelerate launch for the local processes.
        """
        # Set environment variables for Torch distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        # The global rank offset for the node could be node_rank*gpus_on_this_node,
        # but Hugging Face Accelerate typically handles that for each local process
        # if we provide the correct machine_rank. We just define RANK=0 here and
        # rely on accelerate arguments for the actual rank assignment.
        os.environ["RANK"] = str(node_rank)  
        os.environ["WORLD_SIZE"] = str(world_size)

        print(f"[Node rank {node_rank}] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, "
              f"GPUs on node={gpus_on_this_node}, WORLD_SIZE={world_size}")

        # Build the accelerate command:
        # --num_processes = number of GPUs on this node
        # --num_machines = total number of nodes in the job
        # --machine_rank = this node's index
        # The local processes then automatically get local_rank from accelerate.
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(gpus_on_this_node),
            "--num_machines", str(len(gpu_nodes)),
            "--machine_rank", str(node_rank),
            "--main_process_ip", master_addr,
            "--main_process_port", str(master_port),
            "--config_file", CONFIG_YAML, 
            "diffusers/examples/dreambooth/train_dreambooth.py",
            "--pretrained_model_name_or_path", MODEL_NAME,
            "--instance_data_dir", INSTANCE_DIR,
            "--class_data_dir", CLASS_DIR,
            "--output_dir", OUTPUT_DIR,
            "--with_prior_preservation",
            "--prior_loss_weight", "1.0",
            "--instance_prompt", "a photo of xyz eldritch cosmic horror, lovecraftian, tentacles, otherworldly, impossibly large, cosmic scale, ancient god, unspeakable horror",
            "--class_prompt", "a photo of colossal cosmic entity, lovecraftian horror, incomprehensible size, city-sized, tentacles, alien geometry",
            "--resolution", "512",
            "--train_batch_size", "1",
            "--gradient_accumulation_steps", "2",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--learning_rate", "5e-6",
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--num_class_images", "200",
            "--max_train_steps", "800"
        ]
        print(f"[Node rank {node_rank}] Launch command: {' '.join(cmd)}")

        # Run the command. If you want to fail on error, use `check=True`.
        result = subprocess.run(cmd, check=False)
        return result.returncode

    # 6) Launch the tasks on each GPU node with the correct resources.
    #    We'll set num_gpus in Ray to match the gpus_on_this_node so the Ray scheduler
    #    places it properly (and doesn't oversubscribe).
    training_futures = []
    for i, node_info in enumerate(gpu_nodes):
        # The node IP (mostly for debugging, we use the master IP for MASTER_ADDR)
        node_ip = node_info["NodeManagerAddress"]
        gpus_for_this_node = local_gpu_counts[i]

        # Each remote task uses 'resources={}' or 'num_gpus' to ensure
        # Ray schedules them onto the correct node. We'll do a simple approach:
        # We can define a custom resource "node:<ip>" or rely on the GPU count.
        # If we want to ensure it lands exactly on that node, we can define:
        #   .options(resources={f"node:{node_ip}": 1})
        # But let's do a simpler approach:
        train_func = run_accelerate_on_node.options(num_gpus=gpus_for_this_node)
        future = train_func.remote(
            node_rank=i,
            master_addr=master_ip,
            master_port=master_port,
            world_size=total_gpus,
            gpus_on_this_node=gpus_for_this_node
        )
        training_futures.append(future)

    # 7) Wait for all tasks to finish and gather the return codes.
    results = ray.get(training_futures)
    print("Accelerate training exit codes:", results)

    # If any is nonzero, you can raise an error:
    for rc in results:
        if rc != 0:
            raise RuntimeError(f"Accelerate training failed on a node with exit code {rc}")

    print("All distributed training tasks completed successfully.")

if __name__ == "__main__":
    main()
