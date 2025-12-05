import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count

import gymnasium as gym
import numpy as np
import torch
import wandb

import sinergym
from RewardEnergy import ESThermalEnergyReward, rank_based_utilities
from model import HVACPolicy

from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper,
    CSVLogger, WandBLogger, is_wrapped
)
from sinergym.utils.logger import WandBOutputFormat
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3 import SAC

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

NUM_CPUS = cpu_count()
print(f"CPU cores: {NUM_CPUS}")

# ============================================================================
# CONFIGURATION (PHASE 2: ENERGY FINE-TUNING + RANK-BASED)
# ============================================================================

ENV_ID = "Eplus-datacenter_dx-mixed-continuous-v1"
EXPERIMENT_DATE = datetime.today().strftime("%Y-%m-%d_%H%M")
EXPERIMENT_NAME = f"ES_HVAC_RANKBASED_{EXPERIMENT_DATE}"

new_actuators = {
    "Cooling_Setpoint_RL": ("Schedule:Compact", "Schedule Value", "Cooling Setpoints"),
    "East_Zone_Fan_Flow": ("Fan", "Fan Air Mass Flow Rate", "EAST ZONE SUPPLY FAN"),
    "West_Zone_Fan_Flow": ("Fan", "Fan Air Mass Flow Rate", "WEST ZONE SUPPLY FAN"),
}

new_action_space = gym.spaces.Box(
    low=np.array([21.0, 2.5, 2.5], dtype=np.float32),
    high=np.array([26.5, 5.0, 5.0], dtype=np.float32),
    dtype=np.float32,
)

BASELINE_ENERGY_REFERENCE = 881279073.96

reward_parameters = dict(
    temp_name=["east_zone_air_temperature", "west_zone_air_temperature"],
    energy_name="HVAC_electricity_demand_rate",
    baseline_energy_reference=BASELINE_ENERGY_REFERENCE,

    T_min_comfort=18.0,
    T_max_comfort=27.0,

    T_zone1_low=27.0,
    T_zone1_high=27.7,
    T_zone2_high=28.5,

    T_warning=27.7,
    T_danger=28.5,
    T_critical=30.0,

    w1=1.0,
    w2=2.0,
    w3=3.5,
    alpha_zone=0.8,

    lambda_peak=0.0,
    beta_peak=0.0,

    C_min=0.93,
    large_negative=-300.0,

    gamma_T=0.5,
    gamma_E=6.0,

    energy_scale_step=10_000.0,
    penalty_mode="exponential",
    debug=True,
)


env_kwargs = dict(
    reward=ESThermalEnergyReward,
    reward_kwargs=reward_parameters,
    actuators=new_actuators,
    action_space=new_action_space,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_flat_params(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Set model parameters from a flattened 1D tensor."""
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(flat[idx: idx + num].view_as(p))
        idx += num

def verify_checkpoint_integrity(policy: torch.nn.Module, theta: torch.Tensor, 
                               tolerance: float = 1e-5) -> bool:
    """
    Verify that policy parameters match theta within tolerance.
    
    Returns:
        bool: True if difference is below tolerance threshold
    """
    theta_from_policy = get_flat_params(policy).detach().cpu()
    diff = torch.norm(theta_from_policy - theta.cpu()).item()
    
    if diff > tolerance:
        print(f"  ⚠️  WARNING: Policy/theta mismatch! Diff={diff:.2e} (tolerance={tolerance:.2e})")
        return False
    else:
        print(f"  ✓ Checkpoint integrity verified (diff={diff:.2e})")
        return True

def save_emergency_best(policy: torch.nn.Module, best_theta: torch.Tensor, 
                       best_F: float, workspace_path: str, iteration: int):
    """
    Immediately save the best model found so far.
    """
    try:
        emergency_path = os.path.join(workspace_path, "best_so_far.pt")
        
        # Create temporary copy of policy state
        temp_policy_state = policy.state_dict()
        
        # Sync with best_theta
        with torch.no_grad():
            set_flat_params(policy, best_theta.to(policy.fc1.weight.device))
        
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "theta": best_theta.cpu(),
            "best_fitness": best_F,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
        }, emergency_path)
        
        # Restore original policy state
        policy.load_state_dict(temp_policy_state)
        
        print(f"  💾 Emergency best saved: {emergency_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to save emergency best: {e}")

def get_reward_obj(env):
    """Extract reward function object from environment wrapper."""
    try:
        return env.get_wrapper_attr("reward_fn")
    except Exception as e:
        print(f"Error accessing reward_fn: {e}")
        return None

# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def create_env(env_id: str, env_kwargs: dict, experiment_name: str,
               use_wandb: bool = False, worker_id: int = 0):
    """Create and wrap environment with appropriate loggers."""
    if worker_id == 0:
        env_name = experiment_name
    else:
        env_name = f"{experiment_name}_proc_{worker_id}"

    env = gym.make(env_id, env_name=env_name, **env_kwargs)
    env = NormalizeObservation(env)
    env = NormalizeAction(env)

    if worker_id == 0:
        env = LoggerWrapper(env)
        env = CSVLogger(env)
        if use_wandb:
            try:
                env = WandBLogger(
                    env,
                    entity="lucabrunetti2001-n",
                    project_name="AVIS",
                    run_name=experiment_name,
                    group="DatacenterDX_ES_RankBased",
                    tags=["ES", "rank-based", "ES-finetuning", "phase2"],
                    save_code=True,
                )
                print(f"[Worker {worker_id}] W&B logging enabled.")
            except Exception as e:
                print(f"Warning: W&B init failed: {e}")
    return env

# ============================================================================
# EPISODE SIMULATION
# ============================================================================

def _extract_temps_from_info(info):
    """Extract zone temperatures from environment info dictionary."""
    east, west = None, None
    if not isinstance(info, dict):
        return east, west
    
    vars_dict = None
    if "variables" in info and isinstance(info["variables"], dict):
        vars_dict = info["variables"]
    elif "sinergym" in info and isinstance(info["sinergym"], dict):
        if "variables" in info["sinergym"]:
            vars_dict = info["sinergym"]["variables"]
    
    if vars_dict:
        east = vars_dict.get("east_zone_air_temperature")
        west = vars_dict.get("west_zone_air_temperature")
    
    return east, west

def run_episode(env, reward_obj, policy: HVACPolicy, device: str = "cpu",
                log_details: bool = False, log_every: int = 0):
    """
    Run a single episode with the given policy.
    
    Returns:
        F_total: Episode fitness score
        metrics: Dictionary of episode metrics
    """
    obs, _ = env.reset()
    reward_obj.reset_episode_buffers()
    
    done = False
    truncated = False
    policy.eval()
    
    total_step_reward = 0.0
    steps = 0
    setpoints, east_temps, west_temps = [], [], []
    
    while not (done or truncated):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(obs_t).squeeze(0).cpu().numpy()
        
        setpoints.append(float(action[0]))
        obs, r, done, truncated, info = env.step(action)
        total_step_reward += float(r)
        steps += 1
        
        east, west = _extract_temps_from_info(info)
        if east is not None:
            east_temps.append(east)
        if west is not None:
            west_temps.append(west)
        
        if log_details and log_every > 0 and (steps % log_every == 0):
            print(f"[STEP {steps:6d}] setpoint={action[0]:5.2f} | "
                  f"east={east if east else 'NA'} | west={west if west else 'NA'}")
    
    F_total, metrics = reward_obj.compute_episode_fitness()
    metrics["steps"] = steps
    metrics["sum_step_reward"] = total_step_reward
    
    if log_details:
        mean_sp = float(np.mean(setpoints)) if setpoints else float("nan")
        mean_east = float(np.mean(east_temps)) if east_temps else float("nan")
        mean_west = float(np.mean(west_temps)) if west_temps else float("nan")
        
        print("\n" + "="*60)
        print("EPISODE SUMMARY")
        print(f"  Steps          : {steps}")
        print(f"  Mean setpoint  : {mean_sp:.2f}")
        print(f"  Mean east temp : {mean_east:.2f}")
        print(f"  Mean west temp : {mean_west:.2f}")
        print(f"  Comfort C      : {metrics.get('C', 0):.3f}")
        print(f"  Tmax           : {metrics.get('Tmax', 0):.2f}")
        print(f"  Esaving        : {metrics.get('Esaving', 0)}")
        print(f"  F_temp         : {metrics.get('F_temp', 0)}")
        print(f"  F_energy       : {metrics.get('F_energy', 0)}")
        print(f"  F_total        : {F_total:.3f}")
        print("="*60 + "\n")
    
    return F_total, metrics

# ============================================================================
# PARALLEL WORKER (PERSISTENT)
# ============================================================================

_worker_env = None

def evaluate_candidate(args):
    """
    Worker function for parallel evaluation of ES candidates.
    Maintains persistent environment to avoid repeated initialization.
    """
    candidate_idx, eps_numpy, theta_numpy, sigma, env_config = args
    global _worker_env
    
    try:
        # Initialize environment once per worker process
        if _worker_env is None:
            pid = os.getpid()
            _worker_env = create_env(
                env_id=env_config["env_id"],
                env_kwargs=env_config["env_kwargs"],
                experiment_name=env_config["experiment_name"],
                use_wandb=False,
                worker_id=pid,
            )
        
        # Create local policy for this candidate
        local_policy = HVACPolicy(
            obs_dim=env_config["obs_dim"],
            action_dim=env_config["action_dim"],
            action_low=env_config["action_low"],
            action_high=env_config["action_high"],
        )
        
        # Apply perturbation: theta' = theta + sigma * epsilon
        eps = torch.from_numpy(eps_numpy).float()
        theta = torch.from_numpy(theta_numpy).float()
        set_flat_params(local_policy, theta + sigma * eps)
        
        reward_obj = get_reward_obj(_worker_env)
        
        fitness, metrics = run_episode(_worker_env, reward_obj, local_policy, 
                                       device="cpu", log_details=False)
        
        return fitness, metrics
    
    except Exception as e:
        print(f"Worker process {os.getpid()} error: {e}")
        if _worker_env is not None:
            try:
                _worker_env.close()
            except:
                pass
            _worker_env = None
        return -1e9, {}

# ============================================================================
# ES TRAINING WITH MIRRORING + ADAM + RANK-BASED FITNESS SHAPING
# ============================================================================

def es_train_rank_based(
    env,
    policy,
    device="cpu",
    num_iterations: int = 75,
    population_size: int = 32,
    sigma: float = 0.006,
    alpha: float = 0.003,
    sigma_decay: float = 0.995,
    num_workers: int = None,
    checkpoint_freq: int = 5,
    use_rank_fitness: bool = True,
):
    """
    Train policy using Evolution Strategies with rank-based fitness shaping.
    
    Features:
    - Mirrored sampling for variance reduction
    - Adam optimizer for parameter updates
    - Rank-based fitness transformation
    - Persistent worker processes for efficiency
    - Checkpointing and early stopping
    """
    if num_workers is None:
        num_workers = max(1, NUM_CPUS - 1)
    if population_size % 2 != 0:
        raise ValueError("population_size must be even for mirrored sampling.")

    theta_init = get_flat_params(policy).detach().cpu()
    theta = torch.nn.Parameter(theta_init.clone())
    optimizer = torch.optim.Adam([theta], lr=alpha)

    best_F = -1e9
    best_theta = theta_init.clone()
    no_improvement_count = 0
    patience = 30
    
    # Initialize logger if WandB is available
    logger = None
    if is_wrapped(env, WandBLogger):
        print("Initializing SB3Logger with WandB...")
        logger = SB3Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(sys.stdout, max_length=120),
                WandBOutputFormat(),
            ],
        )
    
    # Configuration for worker processes
    env_config = {
        "env_id": ENV_ID,
        "env_kwargs": env_kwargs,
        "experiment_name": EXPERIMENT_NAME,
        "obs_dim": policy.fc1.in_features,
        "action_dim": policy.fc_out.out_features,
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
    }
    
    try:
        workspace_path = env.get_wrapper_attr("workspace_path")
    except Exception:
        workspace_path = f"./experiments/{EXPERIMENT_NAME}"
    os.makedirs(workspace_path, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ES TRAINING - PERSISTENT WORKERS - "
          f"{'RANK-BASED' if use_rank_fitness else 'STANDARD'} FITNESS SHAPING")
    print(f"{'='*70}\n")
    
    with Pool(processes=num_workers) as pool:
        
        for it in range(1, num_iterations + 1):
            with torch.no_grad():
                theta_cpu = theta.detach().cpu()
            
            # Generate mirrored noise samples
            half = population_size // 2
            base_noises = [torch.randn_like(theta_cpu) for _ in range(half)]
            noises = base_noises + [ -n for n in base_noises ]
            
            # Prepare arguments for parallel evaluation
            worker_args = [
                (i + 1, noises[i].numpy(), theta_cpu.numpy(), sigma, env_config)
                for i in range(population_size)
            ]
            
            # Evaluate all candidates in parallel
            results = pool.map(evaluate_candidate, worker_args)
            
            fitnesses = torch.tensor([r[0] for r in results], dtype=torch.float32)
            all_metrics = [r[1] for r in results]
            
            mean_F = fitnesses.mean().item()
            std_F = fitnesses.std().item()
            max_F = fitnesses.max().item()
            
            # Apply fitness shaping (rank-based or standard normalization)
            if use_rank_fitness:
                utilities = rank_based_utilities(fitnesses.numpy())
                print(f"[Iter {it:3d}] Using RANK-BASED utilities (mean={utilities.mean():.3f}, std={utilities.std():.3f})")
            else:
                if std_F > 1e-8:
                    utilities = (fitnesses - mean_F) / (std_F + 1e-8)
                else:
                    utilities = torch.zeros_like(fitnesses)
                print(f"[Iter {it:3d}] Using STANDARD normalization")
            
            # Compute gradient estimate
            grad = torch.zeros_like(theta_cpu)
            for u, eps in zip(utilities, noises):
                grad += u * eps
            grad /= (population_size * sigma)
            
            # Gradient clipping
            raw_grad_norm = torch.norm(grad).item()
            if raw_grad_norm > 10.0:
                grad = grad * (10.0 / raw_grad_norm)
            grad_norm = torch.norm(grad).item()
            
            # Update parameters using Adam
            optimizer.zero_grad()
            theta.grad = -grad.to(theta.dtype)
            optimizer.step()
            
            with torch.no_grad():
                theta_detached = theta.detach().cpu()
            
            # Track best solution
            improvement = False
            improvement_delta = 0.0
            if mean_F > best_F:
                improvement_delta = mean_F - best_F
                best_F = mean_F
                best_theta = theta_detached.clone()
                no_improvement_count = 0
                improvement = True
                
                # Save best model immediately
                save_emergency_best(policy, best_theta, best_F, workspace_path, it)
            else:
                no_improvement_count += 1
            
            # Adaptive sigma adjustment
            if std_F < 0.01 * abs(mean_F) and mean_F > -900:
                sigma *= 1.05
            else:
                sigma *= sigma_decay
            
            # Log metrics to WandB
            if logger is not None and all_metrics:
                mean_C = np.mean([m.get("C", 0) for m in all_metrics])
                mean_S_zone = np.mean([m.get("S_zone", 0) for m in all_metrics])
                mean_Tmax = np.mean([m.get("Tmax", 0) for m in all_metrics])
                mean_Esaving = np.mean([
                    m.get("Esaving", 0) if m.get("Esaving") is not None else 0
                    for m in all_metrics
                ])
                mean_F_temp = np.mean([
                    m.get("F_temp", 0) if m.get("F_temp") is not None else 0
                    for m in all_metrics
                ])
                mean_F_energy = np.mean([
                    m.get("F_energy", 0) if m.get("F_energy") is not None else 0
                    for m in all_metrics
                ])
                
                logger.record("train/iteration", it)
                logger.record("train/sigma", sigma)
                logger.record("train/grad_norm_raw", raw_grad_norm)
                logger.record("train/grad_norm_clipped", grad_norm)
                logger.record("train/fitness_shaping", "rank-based" if use_rank_fitness else "standard")
                logger.record("train/no_improvement_count", no_improvement_count)
                
                logger.record("fitness/mean", mean_F)
                logger.record("fitness/best_current", max_F)
                logger.record("fitness/best_global", best_F)
                logger.record("fitness/std", std_F)
                if improvement:
                    logger.record("fitness/improvement_delta", improvement_delta)
                
                logger.record("metrics/comfort_rate_C", mean_C)
                logger.record("metrics/severity_zone", mean_S_zone)
                logger.record("metrics/temp_max", mean_Tmax)
                logger.record("metrics/energy_saving", mean_Esaving)
                
                logger.record("components/F_temp", mean_F_temp)
                logger.record("components/F_energy", mean_F_energy)
                
                logger.dump(step=it)
            
            # Print iteration summary
            mean_C_log = np.mean([m.get("C", 0) for m in all_metrics]) if all_metrics else 0.0
            improvement_str = f"✓ NEW BEST (+{improvement_delta:.2f})" if improvement else ""
            print(
                f"[Iter {it:3d}] mean={mean_F:7.2f} | max={max_F:7.2f} | "
                f"best={best_F:7.2f} | σ={sigma:.3f} | C={mean_C_log:.2f} "
                f"| grad_norm={grad_norm:5.2f} {improvement_str}"
            )
            
            # Save checkpoint periodically
            if it % checkpoint_freq == 0 or it == num_iterations:
                print(f"\n[CHECKPOINT] Saving iteration {it}...")
                
                # Sync policy with best_theta
                with torch.no_grad():
                    set_flat_params(policy, best_theta.to(policy.fc1.weight.device))
                
                # Verify checkpoint integrity before saving
                is_valid = verify_checkpoint_integrity(policy, best_theta, tolerance=1e-5)
                
                cp_path = os.path.join(workspace_path, f"checkpoint_iter{it:04d}.pt")
                torch.save({
                    "theta": best_theta.cpu(),
                    "best_fitness": best_F,
                    "sigma": sigma,
                    "alpha": alpha,
                    "iteration": it,
                    "policy_state_dict": policy.state_dict(),
                    "use_rank_fitness": use_rank_fitness,
                    "no_improvement_count": no_improvement_count,
                    "checkpoint_valid": is_valid,
                    "timestamp": datetime.now().isoformat(),
                }, cp_path)
                print(f"  → Checkpoint saved: {cp_path}\n")
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"\n⚠️  Early stopping at iteration {it} (no improvement for {patience} iters)")
                break

    # Final synchronization with best parameters
    print("\n[FINAL] Syncing policy with best_theta...")
    with torch.no_grad():
        set_flat_params(policy, best_theta.to(policy.fc1.weight.device))
    
    # Final verification
    verify_checkpoint_integrity(policy, best_theta, tolerance=1e-5)
    
    return best_theta, best_F

# ============================================================================
# SAC INITIALIZATION (NOT FROM ES CHECKPOINT)
# ============================================================================

def load_sac_initialization(policy: HVACPolicy, device: str = "cpu") -> bool:
    """
    Initialize ES policy with pre-trained SAC weights.

    Loading order:
    1) sac_actor_weights_RESTRICTED.pt   (direct cache for HVACPolicy)
    2) sac_warmup_new_ranges.zip         (SAC SB3 model, weights are mapped)
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    actor_weights_path = "sac_actor_weights_RESTRICTED.pt"
    if os.path.exists(actor_weights_path):
        print(f"\n[INIT] Loading SAC actor weights from '{actor_weights_path}'...")
        state_dict = torch.load(actor_weights_path, map_location=device)
        policy.load_state_dict(state_dict, strict=False)
        print("[INIT] SAC weights loaded into HVACPolicy.\n")
        return True

    sac_model_path = "sac_warmup_new_ranges.zip"
    if os.path.exists(sac_model_path) or os.path.exists(sac_model_path + ".zip"):
        print(f"\n[INIT] Loading SAC model from '{sac_model_path}'...")
        try:
            sac_model = SAC.load(sac_model_path, device=device)
            sb3_weights = sac_model.policy.actor.state_dict()

            # Map SB3 SAC actor weights to HVACPolicy architecture
            es_weights = {
                "fc1.weight": sb3_weights["latent_pi.0.weight"],
                "fc1.bias":   sb3_weights["latent_pi.0.bias"],
                "fc2.weight": sb3_weights["latent_pi.2.weight"],
                "fc2.bias":   sb3_weights["latent_pi.2.bias"],
                "fc_out.weight": sb3_weights["mu.weight"],
                "fc_out.bias":   sb3_weights["mu.bias"],
            }

            policy.load_state_dict(es_weights, strict=False)
            torch.save(es_weights, actor_weights_path)
            print(f"[INIT] SAC weights mapped & cached to '{actor_weights_path}'.\n")
            return True
        except Exception as e:
            print(f"[INIT] Error loading SAC model: {e}")

    print("[INIT] No SAC initialization found. Starting from random parameters.\n")
    return False

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ES FINETUNING FROM PRETRAINED SAC (PERSISTENT WORKERS, RANK-BASED)")
    print("="*70 + "\n")
    
    # Create main environment with WandB logging
    env = create_env(
        env_id=ENV_ID,
        env_kwargs=env_kwargs,
        experiment_name=EXPERIMENT_NAME,
        use_wandb=True,
        worker_id=0,
    )
    
    # Get reward object from environment
    reward_obj = get_reward_obj(env)
    if reward_obj is None:
        print("FATAL: Reward object not found.")
        sys.exit(1)
    
    # Initialize policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = HVACPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim
    ).to(device)
    
    # Warm-start from SAC instead of ES checkpoint
    _ = load_sac_initialization(policy, device=device)
    
    # Run debug episode with detailed logging
    print("\n[TEST] Running debug episode with detailed logging...\n")
    _ = run_episode(env, reward_obj, policy, device=device, 
                    log_details=True, log_every=1000)
    
    # Start ES training
    try:
        best_theta, best_F = es_train_rank_based(
            env=env,
            policy=policy,
            device=device,
            num_iterations=75,
            population_size=32,
            sigma=0.045,
            alpha=0.003,
            use_rank_fitness=True,
        )
        
        workspace_path = env.get_wrapper_attr("workspace_path")
        
        # Final guaranteed synchronization
        print("\n[FINAL SAVE] Syncing and saving final model...")
        with torch.no_grad():
            set_flat_params(policy, best_theta.to(device))
        
        # Final verification
        verify_checkpoint_integrity(policy, best_theta, tolerance=1e-5)
        
        # Save final model
        final_path = os.path.join(workspace_path, "final_policy_rank_based.pt")
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "theta": best_theta.cpu(),
            "best_fitness": best_F,
            "timestamp": datetime.now().isoformat(),
        }, final_path)
        
        print(f"\n✓ Training complete. Best fitness: {best_F:.2f}")
        print(f"✓ Final model saved: {final_path}")
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        if wandb.run is not None:
            wandb.finish()
            print("\n✓ WandB run closed.")