import numpy as np
from scipy.special import gamma
from optimizer import Optimizer

# Add PyTorch imports for KLM functionality
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Dict, List, Tuple
from utils import Target
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from collections import deque

class OriginalPLO(Optimizer):

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def levy(self, d):
        """
        Levy flight distribution

        Args:
            d (int): Dimension of the problem
        """
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = self.generator.normal(0, sigma, d)
        v = self.generator.normal(0, 1, d)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate progress percentage for adaptive parameters
        progress_ratio = epoch / self.epoch

        # Calculate mean position of the population
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)

        # Calculate adaptive weights
        w1 = np.tanh((progress_ratio) ** 4)
        w2 = np.exp(-(2 * progress_ratio) ** 3)

        # E for particle collision
        E = np.sqrt(progress_ratio)

        # Generate random permutation for collision pairs
        A = self.generator.permutation(self.pop_size)

        pop_new = []
        for idx in range(0, self.pop_size):
            # Aurora oval walk
            a = self.generator.uniform() / 2 + 1
            V = np.exp((1 - a) / 100 * epoch)
            LS = V

            # Levy flight movement component
            GS = self.levy(self.problem.n_dims) * (x_mean - self.pop[idx].solution +
                                                   (self.problem.lb + self.generator.uniform(0, 1, self.problem.n_dims) *
                                                    (self.problem.ub - self.problem.lb)) / 2)

            # Update position based on aurora oval walk
            pos_new = self.pop[idx].solution + (w1 * LS + w2 * GS) * self.generator.uniform(0, 1, self.problem.n_dims)

            # Particle collision
            for j in range(self.problem.n_dims):
                if (self.generator.random() < 0.05) and (self.generator.random() < E):
                    pos_new[j] = self.pop[idx].solution[j] + np.sin(self.generator.random() * np.pi) * \
                                 (self.pop[idx].solution[j] - self.pop[A[idx]].solution[j])

            # Ensure the position is within bounds
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class KlmNet(nn.Module):
    def __init__(self, n_dims, hidden_nodes=16):
        super(KlmNet, self).__init__()
        self.n_dims = n_dims
        self.fc1 = nn.Linear(n_dims, hidden_nodes)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_nodes, n_dims)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        direction = self.fc3(x)
        return direction


class KLEPLO(OriginalPLO):
    def __init__(self, epoch=10000, pop_size=100,
                 klm_usage_probability: float = 0.2,
                 klm_training_freq: int = 10,
                 klm_lr: float = 0.001,
                 klm_warmup_epochs: int = 20,
                 klm_max_experiences: int = 1000,
                 klm_experience_batch_size: int = 64,
                 klm_step_scale: float = 1.0,
                 klm_epoch_per_update: int = 10,
                 **kwargs):
        
        super().__init__(epoch, pop_size, **kwargs)

        # New KLM parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.klm_usage_probability = klm_usage_probability
        self.klm_training_freq = klm_training_freq
        self.klm_lr = klm_lr
        self.klm_warmup_epochs = klm_warmup_epochs
        self.klm_max_experiences = klm_max_experiences
        self.klm_experience_batch_size = klm_experience_batch_size
        self.klm_step_scale = klm_step_scale
        self.klm_epoch_per_update = klm_epoch_per_update

        self.klm_net = None
        self.klm_optimizer = None
        self.klm_criterion = nn.MSELoss()
        self.klm_experience_buffer = deque(maxlen=self.klm_max_experiences)
        self.klm_initialized = False
        self.klm_usage_count = 0

    # Initialized KLM
    def _initialize_klm_net(self):
        if not self.problem:
            if self.logger: self.logger.error("Problem not defined. KLM Network cannot be initialized.")
            return
        try:
            self.klm_net = KlmNet(self.problem.n_dims).to(self.device)
            self.klm_optimizer = torch.optim.Adam(self.klm_net.parameters(), lr=self.klm_lr)
            self.klm_initialized = True
            if self.logger: self.logger.info(f"KLM Network initialized on {self.device}")
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to initialize KLM Network: {e}")
            self.klm_initialized = False

    # Collect successful experiences
    def _collect_klm_experience(self, original_solution: np.ndarray, successful_new_solution: np.ndarray):
        direction = successful_new_solution - original_solution
        if np.linalg.norm(direction) > self.EPSILON:
            self.klm_experience_buffer.append({
                'position': original_solution.copy(),
                'direction': direction.copy()
            })

    # Train the KLM
    def _train_klm_net(self):
        if len(self.klm_experience_buffer) < self.klm_experience_batch_size or not self.klm_initialized:
            return

        try:
            self.klm_net.train()
            total_loss_epoch = 0
            batches_processed = 0

            for _ in range(self.klm_epoch_per_update):
                indices = self.generator.choice(len(self.klm_experience_buffer), self.klm_experience_batch_size,
                                                replace=len(self.klm_experience_buffer) < self.klm_experience_batch_size)
                batch = [self.klm_experience_buffer[idx] for idx in indices]

                positions = np.array([exp['position'] for exp in batch])
                target_directions = np.array([exp['direction'] for exp in batch])

                X = torch.FloatTensor(positions).to(self.device)
                y_directions = torch.FloatTensor(target_directions).to(self.device)

                self.klm_optimizer.zero_grad()
                predicted_directions = self.klm_net(X)
                loss = self.klm_criterion(predicted_directions, y_directions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.klm_net.parameters(), max_norm=1.0)
                self.klm_optimizer.step()
                total_loss_epoch += loss.item()
                batches_processed += 1

            if batches_processed > 0 and self.logger:
                avg_loss = total_loss_epoch / batches_processed
                self.logger.info(f"KLM Net training: Avg Loss:  {avg_loss:.4e}")

            self.klm_net.eval()

        except Exception as e:
            if self.logger: self.logger.warning(f"KLM network training error: {e}")

    # Utilize KLM to guide evolution
    def _apply_klm_guidance(self, current_agent_idx: int) -> np.ndarray:
        if not self.klm_initialized or self.klm_net is None:
            return self.pop[current_agent_idx].solution.copy()

        current_pos = self.pop[current_agent_idx].solution

        try:
            self.klm_net.eval()
            with torch.no_grad():
                pos_tensor = torch.FloatTensor(current_pos).to(self.device)
                predicted_direction = self.klm_net(pos_tensor).squeeze().cpu().numpy()

            movement = self.klm_step_scale * self.generator.uniform(0.5, 1.5) * predicted_direction
            new_solution = current_pos + movement
            return self.correct_solution(new_solution)

        except Exception as e:
            if self.logger: self.logger.error(f"KLM guidance error: {e}")
            return current_pos.copy()

    def evolve(self, epoch: int):
        # Initialize KLM net if not done and warmup passed
        if not self.klm_initialized and epoch >= self.klm_warmup_epochs // 2:
            self._initialize_klm_net()

        progress_ratio = epoch / self.epoch

        # Original PLO calculations
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)
        w1 = np.tanh(progress_ratio ** 4)
        w2 = np.exp(-(2 * progress_ratio) ** 3)
        E = np.sqrt(progress_ratio)
        A = self.generator.permutation(self.pop_size)

        newly_generated_solutions_data = []

        for i in range(self.pop_size):
            current_agent_original_solution = self.pop[i].solution.copy()
            new_sol_vector = None
            operator_used = "plo"

            rand_val = self.generator.random()

            if rand_val < self.klm_usage_probability and epoch \
                        >= self.klm_warmup_epochs and len(self.klm_experience_buffer) \
                        >= self.klm_experience_batch_size:
                new_sol_vector = self._apply_klm_guidance(i)
                operator_used = "klm"
                self.klm_usage_count += 1
            else:
                a = self.generator.uniform() / 2 + 1.0
                V_local_search_factor = np.exp((1 - a) / (100.0 + self.EPSILON) * epoch)
                GS_levy = self.levy(self.problem.n_dims) * \
                          (x_mean - current_agent_original_solution +
                          (self.problem.lb + self.generator.uniform(0, 1, self.problem.n_dims) *
                          (self.problem.ub - self.problem.lb)) / 2)
                plo_movement = (w1 * V_local_search_factor * (x_mean - current_agent_original_solution) +
                                w2 * GS_levy) * \
                                self.generator.uniform(0, 1, self.problem.n_dims)
                new_sol_vector = current_agent_original_solution + plo_movement
                new_sol_vector = self.correct_solution(new_sol_vector)

            newly_generated_solutions_data.append({
                'original_idx': i,
                'original_solution': current_agent_original_solution,
                'new_solution': new_sol_vector,
                'old_fitness': self.pop[i].target.fitness,
                'operator_used': operator_used
            })

        # Evaluate and select, and collect KLM experiences
        for data in newly_generated_solutions_data:
            idx = data['original_idx']
            original_solution_for_exp = data['original_solution']
            new_solution = data['new_solution']
            original_fitness = data['old_fitness']

            new_agent = self.generate_empty_agent(new_solution)
            new_agent.target = self.get_target(new_solution)

            success = False
            if self.problem.minmax == "min":
                if new_agent.target.fitness < original_fitness: success = True
            else:
                if new_agent.target.fitness > original_fitness.target.fitness: success = True

            # Universal KLM experience collection
            if success and self.klm_initialized:
                self._collect_klm_experience(original_solution_for_exp, new_agent.solution)

            self.pop[idx] = self.get_better_agent(self.pop[idx], new_agent, self.problem.minmax)

        # Train KLM Net
        if epoch >= self.klm_warmup_epochs and epoch % self.klm_training_freq == 0 and self.klm_initialized:
            self._train_klm_net()



class FitnessNet(nn.Module):
    """Simple network to predict fitness values from solutions"""
    def __init__(self, inputs, hidden_nodes=32, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_nodes, hidden_nodes // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_nodes // 2, 1)  # Single output for fitness
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Remove last dimension

class DirNet(nn.Module):
    def __init__(self, inputs, hidden_nodes, outputs, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_nodes),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Configurable dropout rate
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_nodes, outputs)
        )

    def forward(self, x):
        return self.net(x)


class FitnessDataset(Dataset):
    """Simple dataset for fitness prediction"""

    def __init__(self, data, lb, ub, fitness_stats=None):
        self.data_list = data
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

        # Position normalization parameters (for features)
        self.pos_scale = ((self.ub - self.lb) / 2.0).clone().detach().to(dtype=torch.float32)
        self.pos_shift = ((self.ub + self.lb) / 2.0).clone().detach().to(dtype=torch.float32)

        # Fitness normalization parameters (for labels)
        if fitness_stats is None:
            # Calculate fitness statistics from the data
            fitness_values = [item['fitness'] for item in data]
            if fitness_values:
                self.fitness_mean = np.mean(fitness_values)
                self.fitness_std = np.std(fitness_values) + 1e-8  # Add small epsilon
            else:
                self.fitness_mean = 0.0
                self.fitness_std = 1.0
        else:
            self.fitness_mean, self.fitness_std = fitness_stats

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        position = torch.tensor(self.data_list[idx]['position'], dtype=torch.float32)
        fitness = self.data_list[idx]['fitness']

        # Normalize position (feature) using position bounds
        norm_position = (position - self.pos_shift) / self.pos_scale

        # Normalize fitness (label) using fitness statistics
        norm_fitness = (fitness - self.fitness_mean) / self.fitness_std

        return norm_position, torch.tensor(norm_fitness, dtype=torch.float32)

    def get_fitness_stats(self):
        """Return fitness normalization statistics"""
        return self.fitness_mean, self.fitness_std

    def denormalize_fitness(self, norm_fitness):
        """Denormalize a fitness value"""
        return norm_fitness * self.fitness_std + self.fitness_mean

class CustomDataset(Dataset):

    def __init__(self, data, lb, ub, pos_stats=None):
        self.data_list = data
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

        # Position normalization parameters (for features)
        self.pos_scale = ((self.ub - self.lb) / 2.0).clone().detach().to(dtype=torch.float32)
        self.pos_shift = ((self.ub + self.lb) / 2.0).clone().detach().to(dtype=torch.float32)

        # Direction normalization parameters (for labels)
        if pos_stats is None:
            # Calculate direction statistics from the data
            directions = [torch.tensor(item['direction'], dtype=torch.float32) for item in data]
            if directions:
                directions_tensor = torch.stack(directions)
                self.dir_mean = directions_tensor.mean(dim=0)
                self.dir_std = directions_tensor.std(dim=0) + 1e-8  # Add small epsilon
            else:
                self.dir_mean = torch.zeros_like(self.pos_shift)
                self.dir_std = torch.ones_like(self.pos_scale)
        else:
            self.dir_mean, self.dir_std = pos_stats

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        feature = torch.tensor(self.data_list[idx]['start_pos'], dtype=torch.float32)
        label = torch.tensor(self.data_list[idx]['direction'], dtype=torch.float32)

        # Normalize position (feature) using position bounds
        norm_feature = (feature - self.pos_shift) / self.pos_scale

        # Normalize direction (label) using direction statistics
        norm_label = (label - self.dir_mean) / self.dir_std

        return norm_feature, norm_label

    def get_direction_stats(self):
        """Return direction normalization statistics"""
        return self.dir_mean, self.dir_std

    def denormalize_direction(self, norm_direction):
        """Denormalize a direction vector"""
        return norm_direction * self.dir_std + self.dir_mean

    
class DirPLO(OriginalPLO):
    def __init__(self, epoch, pop_size, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

        self.dirnet = None
        self.optimizer = None
        self.scheduler = None
        self.batch_size = 256
        self.hidden_nodes = 64
        self.criterion = DirPLO.cosine_loss
        self.writer = None
        self.model_trained = False
        self.device = 'cpu'
        self.train_every = 10
        self.n_grad_epochs = 5
        self.data = []
        self.train_loader = []
        self.val_loader = []
        self.scheduler = None

        # Fixed train/val split to prevent data leakage
        self.fixed_train_indices = None
        self.fixed_val_indices = None
        self.last_split_size = 0

        # Training configuration
        self.dropout_rate = 0.0  # Disable dropout to fix train/val discrepancy
        self.use_weight_decay = True
        self.weight_decay = 1e-5  # Reduced weight decay
        self.learning_rate = 5e-4  # Lower learning rate for stability


    @staticmethod
    def cosine_loss(pred, tgt):
        pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
        tgt = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-12)
        return 1 - (pred * tgt).sum(dim=1).mean()

    @staticmethod
    def mse_loss(pred, tgt):
        """Alternative MSE loss for direction prediction"""
        return torch.nn.functional.mse_loss(pred, tgt)

    @staticmethod
    def combined_loss(pred, tgt, alpha=0.7):
        """Combined cosine and MSE loss for better stability"""
        cosine_loss = DirPLO.cosine_loss(pred, tgt)
        mse_loss = DirPLO.mse_loss(pred, tgt)
        return alpha * cosine_loss + (1 - alpha) * mse_loss

    def close_writer(self):
        """Close the SummaryWriter if it exists"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            self.writer = None

    # Override the method to collect data for model training
    def get_better_agent_with_data(self, agent_x, agent_y, minmax: str = "min", reverse: bool = False):
        """
        Collect training data: direction from worse solution to better solution.
        This creates consistent training signal regardless of minmax setting.

        Args:
            reverse: Unused parameter kept for compatibility with parent class
        """
        # Determine which agent is better based on minmax
        if minmax == "min":
            # For minimization, lower fitness is better
            if agent_x.target.fitness < agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x
        else:  # maxmax == "max"
            # For maximization, higher fitness is better
            if agent_x.target.fitness > agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x

        # Always collect data: from worse position toward better position
        self.data.append({
            'start_pos': worse_agent.solution.copy(),
            'direction': better_agent.solution.copy() - worse_agent.solution.copy()
        })

        return better_agent.copy()

    def evolve(self, epoch):
        # Calculate progress percentage for adaptive parameters
        progress_ratio = epoch / self.epoch

        # Calculate mean position of the population
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)

        # Calculate adaptive weights
        w1 = np.tanh((progress_ratio) ** 4)
        w2 = np.exp(-(2 * progress_ratio) ** 3)

        # E for particle collision
        E = np.sqrt(progress_ratio)

        # Generate random permutation for collision pairs
        A = self.generator.permutation(self.pop_size)

        pop_new = []

        for idx in range(self.pop_size):

            if self.model_trained and self.dirnet is not None:
                # --- ask the net ---
                inp = torch.tensor(self.pop[idx].solution, dtype=torch.float32)
                with torch.no_grad():
                    # Normalize input using position bounds (same as training)
                    pos_scale = ((torch.tensor(self.problem.ub, dtype=torch.float32) - torch.tensor(self.problem.lb, dtype=torch.float32)) / 2.0)
                    pos_shift = ((torch.tensor(self.problem.ub, dtype=torch.float32) + torch.tensor(self.problem.lb, dtype=torch.float32)) / 2.0)
                    norm_inp = (inp - pos_shift) / pos_scale

                    dir_unit = self.dirnet(norm_inp.to(self.device))
                    dir_unit = dir_unit.detach().cpu().numpy()
                    dir_unit /= np.linalg.norm(dir_unit) + 1e-12  # force unit

                if not hasattr(self, 'sigma0'):
                    self.sigma0 = 0.1 * np.linalg.norm(self.problem.ub - self.problem.lb)
                tau = 0.95
                sigma = self.sigma0 * (tau ** epoch)
                pos_new = self.pop[idx].solution + sigma * dir_unit

            else:
                # --- aurora-oval walk + Lévy flight ---------------------------------
                a = self.generator.uniform() / 2 + 1
                V = np.exp((1 - a) / 100 * epoch)
                LS = V
                GS = (self.levy(self.problem.n_dims) *
                      (x_mean - self.pop[idx].solution +
                       (self.problem.lb + self.generator.uniform(0, 1,
                                                                 self.problem.n_dims) *
                        (self.problem.ub - self.problem.lb)) / 2))
                pos_new = (self.pop[idx].solution +
                           (w1 * LS + w2 * GS) *
                           self.generator.uniform(0, 1, self.problem.n_dims))

            # --- optional collision (one *extra* random perturbation) ---------------
            if self.generator.random() < 0.05 and self.generator.random() < E:
                j = self.generator.integers(0, self.problem.n_dims)  # pick one dim
                pos_new[j] += np.sin(self.generator.random() * np.pi) * \
                              (self.pop[idx].solution[j] -
                               self.pop[A[idx]].solution[j])

            # --- clip to bounds & add to new pop ------------------------------------
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop_new.append(agent)


            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent_with_data(agent, self.pop[idx], self.problem.minmax)

        if epoch % self.train_every == 0 and len(self.data) > 1024:
            self.train_dirnet(epoch)

    def _create_fixed_train_val_split(self, data_size):
        """Create a fixed train/validation split that doesn't change between training sessions"""
        if (self.fixed_train_indices is None or
            self.fixed_val_indices is None or
            self.last_split_size != data_size):

            # Create new fixed split
            indices = np.arange(data_size)
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42, shuffle=True
            )
            self.fixed_train_indices = train_indices
            self.fixed_val_indices = val_indices
            self.last_split_size = data_size
            print(f"Created new fixed train/val split: {len(train_indices)} train, {len(val_indices)} val")

        return self.fixed_train_indices, self.fixed_val_indices

    def train_dirnet(self, epoch):
        """Train DirNet periodically during the run."""
        if epoch < self.epoch // 2:
            return                      # Wait until halfway through the run

        try:
            # 1.  Build the data sets ------------------------------------------------
            self.data = self.data[-10240:]

            if len(self.data) < 100:  # Need minimum data for meaningful training
                print(f"Insufficient data for training: {len(self.data)} samples")
                return

            # Use fixed train/val split to prevent data leakage
            train_indices, val_indices = self._create_fixed_train_val_split(len(self.data))
            train = [self.data[i] for i in train_indices]
            val = [self.data[i] for i in val_indices]

            # Create training dataset first to get direction statistics
            train_ds = CustomDataset(train, self.problem.lb, self.problem.ub)
            # Use same direction statistics for validation to prevent data leakage
            dir_stats = train_ds.get_direction_stats()
            val_ds = CustomDataset(val, self.problem.lb, self.problem.ub, pos_stats=dir_stats)

            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                      shuffle=True, drop_last=False)
            self.val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                      shuffle=False, drop_last=False)

            # 2.  Initialize the net only if it doesn't exist ------------------------
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims, dropout_rate=self.dropout_rate).to(self.device)

            # Always create new writer and optimizer for this training session
            self.writer = SummaryWriter('./log/')

            # Use more conservative training settings
            if self.use_weight_decay:
                self.optimizer = optim.AdamW(self.dirnet.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
            else:
                self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Use simpler scheduler to avoid training instability
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)

            # 3.  Train for a few *gradient* epochs, not 100 full passes -------------
            print(f"Training DirNet at epoch {epoch} with {len(train)} train samples, {len(val)} val samples")
            print(f"Train loader batches: {len(self.train_loader)}, Val loader batches: {len(self.val_loader)}")
            print(f"Using dropout: {self.dropout_rate}, weight_decay: {self.weight_decay}, lr: {self.learning_rate}")

            # ─────────────────── train & validate each epoch ───────────────────
            for e in range(self.n_grad_epochs):
                # training loop --------------------------------------------------
                self.dirnet.train()
                train_loss_acc = 0.0
                train_batch_count = 0
                train_cosine_similarities = []

                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward pass
                    pred = self.dirnet(x)
                    loss = self.criterion(pred, y)

                    # Calculate cosine similarity for monitoring (before normalization)
                    with torch.no_grad():
                        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
                        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-12)
                        cosine_sim = (pred_norm * y_norm).sum(dim=1).mean().item()
                        train_cosine_similarities.append(cosine_sim)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.dirnet.parameters(), 1.0)
                    self.optimizer.step()

                    train_loss_acc += loss.item()
                    train_batch_count += 1

                # Step scheduler once per epoch, not per batch
                self.scheduler.step()
                train_loss = train_loss_acc / train_batch_count
                avg_train_cosine = np.mean(train_cosine_similarities)

                # validation loop ------------------------------------------------
                self.dirnet.eval()
                val_loss_acc = 0.0
                val_batch_count = 0
                val_cosine_similarities = []

                with torch.no_grad():
                    for x, y in self.val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.dirnet(x)
                        loss = self.criterion(pred, y)

                        # Calculate cosine similarity for monitoring
                        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
                        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-12)
                        cosine_sim = (pred_norm * y_norm).sum(dim=1).mean().item()
                        val_cosine_similarities.append(cosine_sim)

                        val_loss_acc += loss.item()
                        val_batch_count += 1

                val_loss = val_loss_acc / val_batch_count
                avg_val_cosine = np.mean(val_cosine_similarities)

                print(f"Epoch {e}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"         Train Cosine = {avg_train_cosine:.6f}, Val Cosine = {avg_val_cosine:.6f}")
                print(f"         LR = {self.optimizer.param_groups[0]['lr']:.6e}")

                # log *each* grad‑epoch (smooth curves, unique x‑axis)
                self.writer.add_scalars(f'Loss/{self.problem.name}', {
                    'train': train_loss,
                    'val':   val_loss
                }, epoch * self.n_grad_epochs + e)  # Use unique x-axis across all training sessions

                self.writer.add_scalars(f'CosineSimilarity/{self.problem.name}', {
                    'train': avg_train_cosine,
                    'val':   avg_val_cosine
                }, epoch * self.n_grad_epochs + e)

            # Mark model as trained for the first time (for inference usage)
            if not self.model_trained:
                self.model_trained = True
        finally:
            self.close_writer()
    # ---------------------------------------------------------------------------