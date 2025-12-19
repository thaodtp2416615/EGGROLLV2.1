"""
EGGROLL Complete Training Loop for Translation Model Finetuning

This module provides a complete training pipeline implementing the EGGROLL
algorithm from "Evolution Strategies at Hyperscale" (arXiv:2511.16652).

Based on: https://github.com/ESHyperscale/HyperscaleES/blob/main/llm_experiments/general_do_evolution_multi_gpu.py

Training Flow:
    Step 1: Initialization (load model, setup noiser)
    Step 2: Generate low-rank perturbations
    Step 3: Forward pass with perturbed models
    Step 4: Compute rewards (BLEU scores)
    Step 5: Estimate gradients using ES
    Step 6: Update parameters
    Step 7: Repeat until convergence
"""

import os
import sys
import csv
import time
import math
import operator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, List, Any, Tuple, Callable
import warnings

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("transformers is required. Install with: pip install transformers")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EggrollTrainerConfig:
    """
    Complete configuration for EGGROLL training.
    Mirrors Args from the original JAX implementation.
    """
    # Random seed
    seed: int = 0
    
    # Model configuration
    model_name: str = "Helsinki-NLP/opus-mt-en-vi"
    
    # Output directories
    save_path: str = "./checkpoints"
    load_path: Optional[str] = None
    
    # Save/Load options
    save_model: bool = True
    load_model: bool = False
    
    # Generation settings
    num_beams: int = 1
    
    # EGGROLL core hyperparameters
    sigma: float = 1e-3              # Noise standard deviation (σ)
    lr_scale: float = 1.0            # Learning rate (α)
    rank: int = 16                   # Low-rank dimension (r)
    noise_reuse: int = 1             # Reuse noise across epochs
    freeze_nonlora: bool = True      # Freeze non-LoRA parameters
    
    # Population settings
    generations_per_prompt: int = 8  # N: population size per unique prompt
    prompts_per_epoch: int = 8       # Number of unique prompts per epoch
    
    # Training settings
    num_epochs: int = 50
    validate_every: int = 200
    save_every: int = 1000
    log_every: int = 10
    
    # Validation settings
    validation_samples: int = 100
    
    # Optimizer settings
    optimizer_type: str = "sgd"      # "sgd" or "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    momentum: float = 0.0
    
    # Reward settings
    reward_metric: str = "bleu"      # "bleu", "meteor", "chrf", "composite"
    # Matches HyperscaleES EggRoll.convert_fitnesses by default
    # Options: "none", "standardize", "centered_rank", "hyperscalees"
    fitness_shaping: str = "hyperscalees"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    
    @property
    def total_generations_per_epoch(self) -> int:
        return self.generations_per_prompt * self.prompts_per_epoch
    


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrainingStats:
    """Statistics for a single epoch"""
    epoch: int
    
    # Fitness statistics
    avg_fitness: float = 0.0
    std_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    median_fitness: float = 0.0
    
    # Update statistics
    lora_param_diff: float = 0.0
    full_param_diff: float = 0.0
    gradient_norm: float = 0.0
    
    # Timing
    prompt_time: float = 0.0
    generation_time: float = 0.0
    fitness_time: float = 0.0
    update_time: float = 0.0
    validation_time: float = 0.0
    saving_time: float = 0.0
    total_time: float = 0.0
    
    # Validation
    validation_score: Optional[float] = None
    
    # Running averages
    true_train_avg_fitness: float = 0.0


@dataclass
class Checkpoint:
    """Checkpoint data structure"""
    epoch: int
    params: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    es_map: Dict[str, int]
    config: Dict[str, Any]
    stats: Dict[str, float]
    timestamp: str


# ============================================================================
# Random Key Generator (JAX-style)
# ============================================================================

class RandomKeyGenerator:
    """JAX-style random key generator for reproducible noise generation."""
    
    def __init__(self, seed: int):
        self.seed = seed
        
    def fold_in(self, key_id: int) -> 'RandomKeyGenerator':
        new_seed = ((self.seed * 31337) + key_id) % (2**31)
        return RandomKeyGenerator(new_seed)
    
    def split(self, num_keys: int) -> List['RandomKeyGenerator']:
        return [self.fold_in(i) for i in range(num_keys)]


# ============================================================================
# ES Map Types
# ============================================================================

class ESMapType:
    FULL = 0
    LORA = 1
    FROZEN = 2
    NOOP = 3


# ============================================================================
# Optimizer State
# ============================================================================

@dataclass
class OptimizerState:
    step: int = 0
    momentum: Optional[Dict[str, torch.Tensor]] = None
    velocity: Optional[Dict[str, torch.Tensor]] = None


# ============================================================================
# EGGROLL Trainer Class
# ============================================================================

class EggrollTrainer:
    """
    Complete EGGROLL trainer for translation model finetuning.
    
    Implements the full training loop from the paper:
    "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
    """
    
    def __init__(self, config: EggrollTrainerConfig):
        """
        Initialize the EGGROLL trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = config.device
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize components (will be set in setup())
        self.model = None
        self.tokenizer = None
        self.params = None
        self.es_map = None
        self.base_evo_keys = None
        self.opt_state = None
        self.reward_function = None
        
        # Training state
        self.current_epoch = 0
        self.true_train_fitness_sum = 0.0
        self.best_validation_score = -float('inf')
        
        # Timing
        self.start_time = None
        
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    # ========================================================================
    # Step 1: Initialization
    # ========================================================================
    
    def setup(self):
        """
        Setup the trainer (Step 1: Initialization).
        
        This includes:
        - Loading the pre-trained model
        - Building ES parameter map
        - Initializing noiser parameters
        - Setting up reward function
        - Initializing wandb (if enabled)
        """
        print("=" * 70)
        print("EGGROLL Trainer Setup")
        print("=" * 70)
        
        # 1. Load model and tokenizer
        print("\n[1/6] Loading model and tokenizer...")
        self._load_model()
        
        # 2. Extract parameters and build ES map
        print("\n[2/6] Building ES parameter map...")
        self._build_es_map()
        
        # 3. Initialize random keys
        print("\n[3/6] Initializing random keys...")
        self._init_random_keys()
        
        # 4.Initialize optimizer state
        print("\n[4/6] Initializing optimizer...")
        self._init_optimizer()
        
        # 5.Setup reward function
        print("\n[5/6] Setting up reward function...")
        self._setup_reward_function()
            
        # 7.Load checkpoint if specified
        if self.config.load_model and self.config.load_path:
            print(f"\nLoading checkpoint from: {self.config.load_path}")
            self._load_checkpoint(self.config.load_path)
            
        self._print_setup_summary()
        
    def _load_model(self):
        """Load pre-trained model and tokenizer."""
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set model to eval mode (we don't use gradients)
        self.model.eval()

        # [NEW] Compile model để tối ưu hóa tốc độ trên A100
        # mode="reduce-overhead" rất tốt cho trường hợp loop nhiều như ES
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("  Model compiled with torch.compile!")
        except Exception as e:
            print(f"  Could not compile model: {e}")
        
        # Extract parameters
        self.params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _build_es_map(self):
        """Build ES parameter classification map."""
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ]
        
        self.es_map = {}
        lora_count = 0
        full_count = 0
        frozen_count = 0
        
        for name, param in self.params.items():
            # Freeze embeddings and layer norms
            if "embed" in name.lower():
                self.es_map[name] = ESMapType.FROZEN
                frozen_count += 1
            elif "layer_norm" in name.lower() or "layernorm" in name.lower():
                self.es_map[name] = ESMapType.FROZEN
                frozen_count += 1
            # Biases get full updates (if not frozen)
            elif "bias" in name.lower():
                if self.config.freeze_nonlora:
                    self.es_map[name] = ESMapType.FROZEN
                    frozen_count += 1
                else:
                    self.es_map[name] = ESMapType.FULL
                    full_count += 1
            # Check for LoRA targets (2D weight matrices)
            elif any(target in name.lower() for target in lora_targets) and len(param.shape) == 2:
                self.es_map[name] = ESMapType.LORA
                lora_count += 1
            else:
                if self.config.freeze_nonlora:
                    self.es_map[name] = ESMapType.FROZEN
                    frozen_count += 1
                else:
                    self.es_map[name] = ESMapType.FULL
                    full_count += 1
                    
        print(f"  LoRA parameters: {lora_count}")
        print(f"  Full parameters: {full_count}")
        print(f"  Frozen parameters: {frozen_count}")
        
    def _init_random_keys(self):
        """Initialize random keys for each parameter."""
        master_key = RandomKeyGenerator(self.config.seed)
        self.base_model_key = master_key.fold_in(0)
        self.base_gen_key = master_key.fold_in(1)
        self.base_valid_key = master_key.fold_in(2)
        
        self.base_evo_keys = {
            name: self.base_model_key.fold_in(i)
            for i, name in enumerate(self.params.keys())
        }
        
    def _init_optimizer(self):
        """Initialize optimizer state."""
        self.opt_state = OptimizerState(step=0)
        
        if self.config.optimizer_type == "adam":
            self.opt_state.momentum = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
            self.opt_state.velocity = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
        elif self.config.momentum > 0:
            self.opt_state.momentum = {
                name: torch.zeros_like(p)
                for name, p in self.params.items()
                if self.es_map.get(name, ESMapType.FROZEN) != ESMapType.FROZEN
            }
            
    def _setup_reward_function(self):
        """Setup reward function for evaluation."""
        metric = self.config.reward_metric.lower()
        
        if metric == "bleu":
            try:
                import sacrebleu
                self._sacrebleu = sacrebleu
                self.reward_function = self._compute_bleu
                print(f"  Reward: BLEU (sacrebleu)")
            except ImportError:
                self.reward_function = self._compute_bleu_nltk
                print(f"  Reward: BLEU (nltk)")
        elif metric == "comet":
            try:
                from comet import download_model, load_from_checkpoint
                # Choose your model from Hugging Face Hub
                # model_path = download_model("Unbabel/XCOMET-XL")
                # or for example:
                model_path = download_model("Unbabel/wmt22-comet-da")

                # Load the model checkpoint:
                model_comet = load_from_checkpoint(model_path)

                self._comet = model_comet
                self.reward_function = self._compute_comet
                print(f"  Reward: COMET (Unbabel/wmt22-comet-da)")
            except ImportError:
                print(f"COMET Error")
        elif metric == "length":
            self.reward_function = self._compute_length_ratio
            print(f"  Reward: Length Ratio")
        else:
            self.reward_function = self._compute_bleu_nltk
            print(f"  Reward: BLEU (nltk, fallback)")
        
    def _print_setup_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print(f"""
Configuration:
  Model: {self.config.model_name}
  Device: {self.device}
  
EGGROLL Hyperparameters:
  σ (sigma): {self.config.sigma}
  α (learning rate): {self.config.lr_scale}
  r (rank): {self.config.rank}
  N (population per prompt): {self.config.generations_per_prompt}
  
Training:
  Epochs: {self.config.num_epochs}
  Prompts per epoch: {self.config.prompts_per_epoch}
  Total generations per epoch: {self.config.total_generations_per_epoch}
  Reward metric: {self.config.reward_metric}
""")

    # ========================================================================
    # Step 2 & 3: Perturbation and Forward Pass
    # ========================================================================
    
    def _get_perturbation_seed(self, base_seed: int, epoch: int, thread_id: int) -> int:
        """
        Matches HyperscaleES seeding: fold in (true_epoch, true_thread_idx)
        where true_thread_idx = thread_id // 2 and true_epoch is controlled by noise_reuse.
        """
        true_thread_idx = thread_id // 2

        # HyperscaleES: true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
        if self.config.noise_reuse == 0:
            true_epoch = 0
        else:
            true_epoch = epoch // self.config.noise_reuse

        return ((base_seed * 31337) + true_epoch * 1000 + true_thread_idx) % (2**31)

    def _generate_lora_perturbation(
        self,
        param_shape: Tuple[int, int],
        seed: int,
        sigma_signed: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate low-rank perturbation matrices A and B.

        Matches HyperscaleES get_lora_update_params + callers:
        - Sample a single (a+b, r) matrix with one seed, then split into B (b, r) and A (a, r)
        - Scale A by (sigma_signed / sqrt(rank)); B is unscaled.
        """
        out_features, in_features = param_shape
        rank = self.config.rank

        # CPU generator for reproducibility across CUDA devices.
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # HyperscaleES uses (a+b, r) where param.shape == (a, b)
        lora_params = torch.randn((out_features + in_features, rank), generator=gen, dtype=torch.float32)
        B = lora_params[:in_features]      # (b, r)
        A = lora_params[in_features:]      # (a, r)

        A = A * (sigma_signed / math.sqrt(rank))
        return A.to(self.device), B.to(self.device)
    @torch.no_grad()
    def _generate_with_perturbation(
        self,
        input_ids: torch.Tensor,
        epoch: int,
        thread_id: int,
    ) -> torch.Tensor:
        """
        Apply a single HyperscaleES-style perturbation (thread) to weights, generate, then restore.

        Important: thread_id is global across all parallel generations; antithetic sign is thread_id % 2.
        """
        sigma_signed = self.config.sigma if (thread_id % 2 == 0) else -self.config.sigma
        
        noise_cache = {} 

        # 1. APPLY NOISE
        for name, param in self.model.named_parameters():
            map_type = self.es_map.get(name, ESMapType.FROZEN)
            if map_type == ESMapType.FROZEN:
                continue

            # Lấy seed (đã sửa ở Bước 1 để trả về seed chung cho cặp)
            base_seed = self.base_evo_keys[name].seed
            seed = self._get_perturbation_seed(base_seed, epoch, thread_id)

            if map_type == ESMapType.LORA and len(param.shape) == 2:
                A, B = self._generate_lora_perturbation(param.shape, seed, sigma_signed)
                delta_w = A @ B.T

                param.data.add_(delta_w)
                
                noise_cache[name] = (A, B)
                
            elif map_type == ESMapType.FULL:
                # HyperscaleES: updates = eps * sigma_signed
                gen = torch.Generator(device="cpu").manual_seed(seed)
                eps = torch.randn(param.shape, generator=gen, dtype=torch.float32)
                noise = (eps.to(device=self.device, dtype=param.dtype) * sigma_signed)
                param.data.add_(noise)
                    
                noise_cache[name] = noise

        # 2. GENERATE
        output_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=self.config.num_beams,
            max_new_tokens=128,
            do_sample=False
        )

        # 3. RESTORE WEIGHTS (Đảo ngược thao tác trên)
        for name, param in self.model.named_parameters():
            if name in noise_cache:
                if self.es_map[name] == ESMapType.LORA:
                    A, B = noise_cache[name]
                    delta_w = A @ B.T

                    param.data.sub_(delta_w)
                else:
                    noise = noise_cache[name]
                    param.data.sub_(noise)
        
        del noise_cache
        return output_ids

    
    # ========================================================================
    # Step 4: Reward Computation
    # ========================================================================
    
    def _compute_bleu(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using sacrebleu."""
        try:
            bleu = self._sacrebleu.sentence_bleu(hypothesis, [reference], smooth_method='exp')
            return bleu.score / 100.0
        except Exception as e:
            print(f"_compute_bleu fail because {e}, Fallback: 0.0")
            return 0.0

    def _compute_comet(self, sources: List[str], hypothesis: List[str], reference: List[str]) -> np.ndarray:
        # Data must be in the following format:
        data = [
            {
                "src": src,
                "mt": hyp,
                "ref": ref
            } for src, hyp, ref in zip(sources, hypothesis, reference)
        ]

        if not sources or not hypothesis or not reference:
            return np.zeros(len(sources))
        try:
            model_output = self._comet.predict(data, batch_size=8, gpus=1)
            return np.array(model_output.scores)
        except:
            return np.zeros(len(sources))
    
    def _compute_bleu_nltk(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using nltk."""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()
            if len(hyp_tokens) == 0:
                return 0.0
            return sentence_bleu([ref_tokens], hyp_tokens, 
                               smoothing_function=SmoothingFunction().method1)
        except:
            return 0.0
    
    def _compute_length_ratio(self, hypothesis: str, reference: str) -> float:
        """Compute length ratio reward."""
        if not reference.strip():
            return 0.0
        hyp_len = len(hypothesis.split())
        ref_len = len(reference.split())
        if ref_len == 0:
            return 0.0
        ratio = hyp_len / ref_len
        return max(0.0, 1.0 - abs(ratio - 1.0))
    
    def _compute_rewards(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> np.ndarray:
        """Compute rewards for all hypotheses."""
        metric = self.config.reward_metric.lower()
        if metric == "comet":
            rewards = self.reward_function(sources, hypotheses, references)
        else:
            rewards = np.array([
                self.reward_function(hyp, ref)
                for hyp, ref in zip(hypotheses, references)
            ])
        return rewards
    
    # ========================================================================
    # Step 5: Fitness Shaping
    # ========================================================================
    
    def _shape_fitnesses(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Apply fitness shaping for ES stability.
        
        Matches HyperscaleES EggRoll.convert_fitnesses.
        """
        if self.config.fitness_shaping == "none":
            return raw_scores
            
        elif self.config.fitness_shaping == "standardize":
            mean = np.mean(raw_scores)
            std = np.std(raw_scores) + 1e-8
            return (raw_scores - mean) / std
            
        elif self.config.fitness_shaping == "centered_rank":
            n = len(raw_scores)
            ranks = np.argsort(np.argsort(raw_scores))
            shaped = (ranks.astype(np.float32) + 0.5) / n - 0.5
            return shaped

        elif self.config.fitness_shaping == "hyperscalees":
            group_size = int(self.config.generations_per_prompt)
            denom = math.sqrt(float(np.var(raw_scores, keepdims=True)) + 1e-5)
            if group_size == 0:
                mean = float(np.mean(raw_scores, keepdims=True))
                return (raw_scores - mean) / denom

            if raw_scores.ndim != 1:
                raise ValueError("hyperscalees fitness_shaping expects 1D raw_scores")
            if raw_scores.size % group_size != 0:
                raise ValueError("raw_scores.size must be divisible by generations_per_prompt")
            group_scores = raw_scores.reshape((-1, group_size))
            group_mean = np.mean(group_scores, axis=-1, keepdims=True)
            true_scores = (group_scores - group_mean) / denom
            return true_scores.ravel()

        else:
            raise ValueError(f"Unknown fitness_shaping: {self.config.fitness_shaping}")
    
    # ========================================================================
    # Steps 5 & 6: Gradient Estimation and Update
    # ========================================================================
    
    def _estimate_and_update(
        self,
        shaped_fitnesses: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Matches HyperscaleES EggRoll._do_update semantics:
        - new_grad = mean(scores * updates)
        - returned grad to optimizer = -(new_grad * sqrt(N))
        """
        population_size = len(shaped_fitnesses)
        
        # Kiểm tra sanity check
        if population_size % 2 != 0:
            raise ValueError("Total parallel generations must be EVEN for antithetic sampling.")

        if self.config.sigma <= 0:
            raise ValueError("sigma must be > 0 for ES gradient estimation")

        stats = {}
        new_params = {}
        lora_diff_sum = 0.0
        lora_count = 0
        full_diff_sum = 0.0
        full_count = 0
        total_grad_norm_sq = 0.0

        for name, param in self.params.items():
            map_type = self.es_map.get(name, ESMapType.FROZEN)

            if map_type == ESMapType.FROZEN:
                new_params[name] = param
                continue

            # new_grad in HyperscaleES (pre -sqrt(N) scaling)
            new_grad = torch.zeros_like(param)

            sqrt_n = math.sqrt(population_size)

            base_seed = self.base_evo_keys[name].seed
            for thread_id in range(population_size):
                score = float(shaped_fitnesses[thread_id])
                sigma_signed = self.config.sigma if (thread_id % 2 == 0) else -self.config.sigma
                seed = self._get_perturbation_seed(base_seed, epoch, thread_id)

                if map_type == ESMapType.LORA and len(param.shape) == 2:
                    A, B = self._generate_lora_perturbation(param.shape, seed, sigma_signed)
                    new_grad += (score * (A @ B.T))

                elif map_type == ESMapType.FULL:
                    gen = torch.Generator(device="cpu").manual_seed(seed)
                    eps = torch.randn(param.shape, generator=gen, dtype=torch.float32)
                    updates = eps.to(device=self.device, dtype=param.dtype) * sigma_signed
                    new_grad += (score * updates)

            new_grad /= population_size

            # grad passed into optimizer matches HyperscaleES return value
            grad_for_opt = -(new_grad * sqrt_n)
            update = self._apply_optimizer_step(name, grad_for_opt)
            new_param = param + update
            new_params[name] = new_param
            
            diff = torch.sqrt(torch.mean((new_param - param) ** 2)).item()
            if map_type == ESMapType.LORA:
                lora_diff_sum += diff
                lora_count += 1
            else:
                full_diff_sum += diff
                full_count += 1
            total_grad_norm_sq += torch.norm(grad_for_opt).item() ** 2

        self.params = new_params
        self._update_model_weights()
        stats['gradient_norm'] = math.sqrt(total_grad_norm_sq)
        stats['lora_param_diff'] = lora_diff_sum / max(lora_count, 1)
        stats['full_param_diff'] = full_diff_sum / max(full_count, 1)

        return stats
    def _apply_optimizer_step(
        self,
        name: str,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an Optax-like optimizer step (returns the *update* to be added to params)."""
        lr = self.config.lr_scale
        
        if self.config.optimizer_type == "adam":
            t = self.opt_state.step + 1
            beta1 = self.config.adam_beta1
            beta2 = self.config.adam_beta2
            eps = self.config.adam_eps
            
            m = self.opt_state.momentum.get(name, torch.zeros_like(gradient))
            v = self.opt_state.velocity.get(name, torch.zeros_like(gradient))
            
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            self.opt_state.momentum[name] = m
            self.opt_state.velocity[name] = v
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            return -lr * m_hat / (torch.sqrt(v_hat) + eps)
            
        elif self.config.momentum > 0:
            m = self.opt_state.momentum.get(name, torch.zeros_like(gradient))
            m = self.config.momentum * m + gradient
            self.opt_state.momentum[name] = m
            return -lr * m
            
        else:
            return -lr * gradient
    
    def _update_model_weights(self):
        """Update model weights from params dictionary."""
        state_dict = self.model.state_dict()
        for name, param in self.params.items():
            if name in state_dict:
                state_dict[name] = param
        self.model.load_state_dict(state_dict)
    
    # ========================================================================
    # Single Epoch
    # ========================================================================
    
    def _train_step(
        self,
        batch_data: List[Tuple[str, str]],
        step_idx: int,

    ) -> TrainingStats:
        """
        Execute a single update step on a batch of data.
        """
        stats = TrainingStats(epoch=step_idx)
        
        # Tạm dùng step làm epoch log
        step_start = time.time()
        
                
        # Sample prompts for this epoch
        epoch_samples = batch_data
        
        # Generate with all population members
        # Matches HyperscaleES: each (prompt_idx, member_idx) is a distinct ES thread.
        gen_start = time.time()

        batch_hypotheses: List[str] = []
        batch_references: List[str] = []
        batch_sources: List[str] = []

        for prompt_idx, (source, reference) in enumerate(tqdm(epoch_samples, desc=f"Prompts (Step {step_idx})", leave=False)):
            inputs = self.tokenizer(
                [source],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=128,
            ).to(self.device)

            for member_idx in range(self.config.generations_per_prompt):
                thread_id = (prompt_idx * self.config.generations_per_prompt) + member_idx
                output_ids = self._generate_with_perturbation(
                    inputs["input_ids"],
                    step_idx,
                    thread_id,
                )

                # output_ids has shape (1, seq_len)
                hypothesis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                batch_hypotheses.append(hypothesis)
                batch_references.append(reference)
                batch_sources.append(source)

        stats.generation_time = time.time() - gen_start

        # Compute rewards (Step 4)
        fitness_start = time.time()
        raw_rewards = self._compute_rewards(batch_sources, batch_hypotheses, batch_references)
        
        stats.fitness_time = time.time() - fitness_start
        # Statistics
        stats.avg_fitness = float(np.mean(raw_rewards))
        stats.std_fitness = float(np.std(raw_rewards))
        
        if stats.std_fitness < 1e-9:
            print(f"  [WARNING] Fitness variance is 0. All rewards are identical! -> Increase Sigma or Batch Size.")

        stats.max_fitness = float(np.max(raw_rewards))
        stats.min_fitness = float(np.min(raw_rewards))
        stats.median_fitness = float(np.median(raw_rewards))
        
        # Shape fitnesses (Step 5a)
        shaped_fitnesses = self._shape_fitnesses(raw_rewards)
        
        # Estimate gradients and update (Steps 5b & 6)
        update_start = time.time()
        update_stats = self._estimate_and_update(shaped_fitnesses, step_idx)
        stats.update_time = time.time() - update_start
        
        stats.lora_param_diff = update_stats['lora_param_diff']
        stats.full_param_diff = update_stats['full_param_diff']
        stats.gradient_norm = update_stats['gradient_norm']
        
        # Increment optimizer step
        self.opt_state.step += 1
        
        # Update running average
        self.true_train_fitness_sum += float(np.sum(raw_rewards))
        stats.true_train_avg_fitness = self.true_train_fitness_sum / ((step_idx + 1) * raw_rewards.size)
        
            
        stats.total_time = time.time() - step_start
        
        return stats
    
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    @torch.no_grad()
    def _validate(
        self,
        val_data: List[Tuple[str, str]],
    ) -> float:
        """
        Run validation on provided data.
        
        Uses base model without perturbation.
        """

        # Unpack the tuples into two separate lists
        list1, list2 = zip(*val_data)

        # Convert the zip objects to lists
        sources, references = list(list1), list(list2)

        batch_inputs = self.tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            batch_output_ids = self.model.generate(
                input_ids=batch_inputs["input_ids"],
                num_beams=self.config.num_beams,
            )
        hypothesis = [
            self.tokenizer.decode(output_ids, skip_special_tokens=True)
            for output_ids in batch_output_ids
        ]

        reward = self.reward_function(sources, hypothesis, references)
        return reward.mean()
    
    # ========================================================================
    # Checkpointing
    # ========================================================================
    
    def _save_checkpoint(self, step: int, stats: TrainingStats):
        """Save training checkpoint."""
        ckpt_dir = Path(self.config.save_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'params': {k: v.cpu() for k, v in self.params.items()},
            'opt_state': {
                'step': self.opt_state.step,
                'momentum': {k: v.cpu() for k, v in (self.opt_state.momentum or {}).items()},
                'velocity': {k: v.cpu() for k, v in (self.opt_state.velocity or {}).items()},
            },
            'es_map': self.es_map,
            'config': asdict(self.config),
            'stats': asdict(stats),
            'true_train_fitness_sum': self.true_train_fitness_sum,
            'best_validation_score': self.best_validation_score,
            'timestamp': datetime.now().isoformat(),
        }
        
        ckpt_path = ckpt_dir / f"checkpoint_step_{step:05d}.pt"
        torch.save(checkpoint, ckpt_path)
        
        # Also save as latest
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        self.model.save_pretrained(f"{ckpt_dir}/checkpoint_step_{step:05d}")
        self.tokenizer.save_pretrained(f"{ckpt_dir}/checkpoint_step_{step:05d}")

        self.model.save_pretrained(f"{ckpt_dir}/checkpoint_last")
        self.tokenizer.save_pretrained(f"{ckpt_dir}/checkpoint_last")
        
        print(f"  Checkpoint saved: {ckpt_path}")
        
    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.params = {
            k: v.to(self.device)
            for k, v in checkpoint['params'].items()
        }
        
        self.opt_state.step = checkpoint['opt_state']['step']
        if checkpoint['opt_state']['momentum']:
            self.opt_state.momentum = {
                k: v.to(self.device)
                for k, v in checkpoint['opt_state']['momentum'].items()
            }
        if checkpoint['opt_state']['velocity']:
            self.opt_state.velocity = {
                k: v.to(self.device)
                for k, v in checkpoint['opt_state']['velocity'].items()
            }
            
        self.current_epoch = checkpoint['epoch'] + 1
        self.true_train_fitness_sum = checkpoint.get('true_train_fitness_sum', 0.0)
        self.best_validation_score = checkpoint.get('best_validation_score', -float('inf'))
        
        self._update_model_weights()
        
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # ========================================================================
    # Logging
    # ========================================================================
    
    def _log_epoch(self, stats: TrainingStats):
        """Log epoch statistics."""
        # Console logging
        if stats.epoch % self.config.log_every == 0:
            print(f"\nEpoch {stats.epoch:5d} | "
                  f"Fitness: {stats.avg_fitness:.4f} ± {stats.std_fitness:.4f} | "
                  f"Best: {stats.max_fitness:.4f} | "
                  f"Grad: {stats.gradient_norm:.6f} | "
                  f"Time: {stats.total_time:.2f}s")
            
            if stats.validation_score is not None:
                print(f"           | Validation: {stats.validation_score:.4f} "
                      f"(Best: {self.best_validation_score:.4f})")
    
    # ========================================================================
    # Main Training Loop
    # ========================================================================
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        val_data: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Main training loop.
        
        Args:
            train_data: List of (source, target) pairs
            val_data: Optional validation data
        """
        print("\n" + "=" * 70)
        print("Starting EGGROLL Training")
        print("=" * 70)
        
        self.start_time = time.time()
        
        global_step = 0
        
        # [FIX] Tạo DataLoader để quản lý batching chuẩn xác
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.config.prompts_per_epoch, # Batch size = 64
            shuffle=True,                             # Xáo trộn dữ liệu mỗi Epoch
            collate_fn=lambda x: x,                   # Giữ nguyên format List[Tuple]
            drop_last=True                            # Bỏ batch lẻ cuối cùng để ổn định
        )

        try:
            # Vòng lặp Epoch thực sự (Duyệt hết dataset)
            for epoch in range(self.config.num_epochs):
                print(f"\n=== Starting Epoch {epoch+1}/{self.config.num_epochs} ===")
                
                # Duyệt qua từng batch trong dataset
                for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                    
                    # Gọi hàm train step (đã sửa ở Bước 2)
                    # Truyền global_step để sinh noise khác nhau cho mỗi batch
                    stats = self._train_step(batch_data, global_step)
                    
                    # Log mỗi bước (hoặc mỗi N bước)
                    if global_step % self.config.log_every == 0:
                        print(f" Step {global_step} | Fitness: {stats.avg_fitness:.4f} | Loss/Diff: {stats.lora_param_diff:.6f}")
                    
                    
                    if global_step > 0 and global_step % self.config.save_every == 0:
                        # Save Regular Checkpoint
                        self._save_checkpoint(step=global_step, stats=stats) # Lưu theo step

                    if global_step > 0 and global_step % self.config.validate_every == 0:
                        print(f"\n--> Running Validation at Step {global_step}...")
                        
                        if val_data:
                            # Chỉ lấy subset 1000 câu để validate cho nhanh
                            val_score = self._validate(val_data)
                            print(f"    Validation BLEU: {val_score:.4f}")
                            
                            # Save Best Model
                            if val_score > self.best_validation_score:
                                self.best_validation_score = val_score
                                print(f"Best checkpoint: {global_step}")
                        
                    global_step += 1

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            
        finally:
            # Final save
            if self.config.save_model:
                print("\nSaving final checkpoint...")
                final_stats = TrainingStats(epoch=self.current_epoch - 1)
                self._save_checkpoint(self.current_epoch - 1, final_stats)
                
                
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation score: {self.best_validation_score:.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for EGGROLL training."""
    
    # Example configuration
    config = EggrollTrainerConfig(
        # Model
        model_name="/home/jovyan/nmt-srv-shared/users/binh/grpo_training/transflow/0_Base/en-vi-2.1.10.04-grpo-100k",
        
        # EGGROLL hyperparameters
        sigma=0.01,                  # [USER CONFIG]
        lr_scale=0.001,              # [USER CONFIG]
        rank=32,                     # [USER CONFIG]
        
        # Population
        generations_per_prompt=128,
        prompts_per_epoch=1024,      # [USER CONFIG]
        
        # Training
        num_epochs=50,
        validate_every=200,
        save_every=400,
        log_every=1,
        
        # Optimizer
        optimizer_type="sgd",
        momentum=0.9,
        
        # Reward
        reward_metric="bleu",
        fitness_shaping="hyperscalees",
        
        # Paths
        save_path="/home/jovyan/nmt-srv-shared/users/binh/EGGROLL/checkpointsv2",
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Example training data (source, target pairs)
    src_train = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/train.src", "r", encoding='utf-8').readlines()
    tgt_train = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/train.tgt", "r", encoding='utf-8').readlines()

    src_valid = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/valid.src", "r", encoding='utf-8').readlines()
    tgt_valid = open("/home/jovyan/nmt-srv-shared/users/binh/eggroll_training/dataset/valid.tgt", "r", encoding='utf-8').readlines()

    train_data = []
    valid_data = []
    for src, tgt in tqdm(zip(src_train, tgt_train), desc="Loading train dataset"):
        train_data.append((src.strip(), tgt.strip()))
    for src, tgt in tqdm(zip(src_valid, tgt_valid), desc="Loading valid dataset"):
        valid_data.append((src.strip(), tgt.strip()))

    # Create trainer
    trainer = EggrollTrainer(config)
    
    # Setup
    trainer.setup()
    
    # Train
    trainer.train(train_data, valid_data)


if __name__ == "__main__":
    main()