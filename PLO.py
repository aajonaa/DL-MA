import numpy as np
from scipy.special import gamma
from optimizer import Optimizer

# Add PyTorch imports for KLM functionality
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Logging will be disabled.")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

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

class DirNet(nn.Module):
    """Simple neural network for direction prediction without dropout or weight decay"""
    def __init__(self, inputs, hidden_nodes, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, outputs)
        )

    def forward(self, x):
        return self.net(x)


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
    """Enhanced PLO with neural network direction prediction, data augmentation, and online learning"""

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=0.5,
                 min_data_for_training=256,  # Much lower for small problems
                 train_every=50,  # Back to more frequent training
                 n_grad_epochs=3,  # Reduced from 5 (fewer training epochs)
                 batch_size=32,  # Reasonable batch size
                 hidden_nodes=8,  # Reduced network size
                 learning_rate=1e-3,  # Increased learning rate for faster convergence
                 # Enhanced magnitude adjustment parameters
                 magnitude_strategy='adaptive_multi',  # 'simple', 'de_style', 'pso_style', 'adaptive_multi'
                 base_f_factor=0.5,
                 f_factor_range=(0.2, 0.8),
                 crossover_rate=0.7,
                 use_population_guidance=True,
                 diversity_threshold=0.1,
                 **kwargs):
        super().__init__(epoch, pop_size, **kwargs)

        # Core parameters
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        # self.augmentation_factor = augmentation_factor
        # self.max_augmented_samples = max_augmented_samples
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Enhanced magnitude adjustment parameters
        self.magnitude_strategy = magnitude_strategy
        self.base_f_factor = base_f_factor
        self.f_factor_range = f_factor_range
        self.crossover_rate = crossover_rate
        self.use_population_guidance = use_population_guidance
        self.diversity_threshold = diversity_threshold

        # Neural network components
        self.dirnet = None
        self.optimizer = None
        self.criterion = nn.MSELoss()  # Simple MSE loss for direction prediction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management
        self.data = []
        self.train_loader = None
        self.val_loader = None

        # Statistics for noise generation
        self.noise_std = None

        # Training session tracking for TensorBoard
        self.global_step_counter = 0

        # Performance optimization parameters
        self.early_stopping_patience = 3  # Stop training if no improvement
        self.min_loss_improvement = 1e-5  # Minimum improvement threshold
        self.last_training_loss = float('inf')
        self.no_improvement_count = 0

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

    def _calculate_population_diversity(self):
        """Calculate population diversity for adaptive magnitude scaling"""
        if len(self.pop) < 2:
            return 1.0

        positions = np.array([agent.solution for agent in self.pop])
        mean_pos = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - mean_pos) for pos in positions]
        avg_distance = np.mean(distances)

        # Normalize by problem bounds
        bounds_range = np.linalg.norm(np.array(self.problem.ub) - np.array(self.problem.lb))
        normalized_diversity = avg_distance / (bounds_range + 1e-12)

        return normalized_diversity

    def _get_adaptive_f_factor(self, epoch, diversity):
        """Calculate adaptive F factor based on epoch progress and population diversity"""
        progress_ratio = epoch / self.epoch

        # Base F factor decreases over time
        time_factor = self.base_f_factor * (1.0 - 0.5 * progress_ratio)

        # Increase F when diversity is low (exploration needed)
        diversity_factor = 1.0 + (1.0 - diversity) * 0.5

        # Apply range constraints
        f_factor = time_factor * diversity_factor
        f_factor = np.clip(f_factor, self.f_factor_range[0], self.f_factor_range[1])

        return f_factor

    def _apply_de_style_magnitude(self, predicted_direction, current_solution, epoch, agent_idx):
        """Apply DE-style magnitude scaling with multiple strategies"""
        diversity = self._calculate_population_diversity()
        f_factor = self._get_adaptive_f_factor(epoch, diversity)

        # Choose DE strategy based on diversity and progress
        progress_ratio = epoch / self.epoch
        strategy_choice = self.generator.random()

        if strategy_choice < 0.3:  # DE/rand/1 style
            # Use predicted direction as primary difference vector
            base_solution = current_solution
            mutant = base_solution + f_factor * predicted_direction

        elif strategy_choice < 0.6:  # DE/best/1 style
            # Combine with direction toward best solution
            best_agent = min(self.pop, key=lambda x: x.target.fitness if self.problem.minmax == "min"
                           else lambda x: -x.target.fitness)
            best_direction = best_agent.solution - current_solution

            # Mix predicted direction with best direction
            alpha = 0.7  # Weight for predicted direction
            combined_direction = alpha * predicted_direction + (1 - alpha) * best_direction
            mutant = current_solution + f_factor * combined_direction

        else:  # DE/current-to-best/1 style
            # Multi-vector approach
            best_agent = min(self.pop, key=lambda x: x.target.fitness if self.problem.minmax == "min"
                           else lambda x: -x.target.fitness)
            best_direction = best_agent.solution - current_solution

            # Use predicted direction as second difference vector
            f1 = f_factor * 0.8  # Slightly reduce for stability
            f2 = f_factor * 1.2

            mutant = (current_solution +
                     f1 * best_direction +
                     f2 * predicted_direction)

        return mutant

    def _apply_pso_style_magnitude(self, predicted_direction, current_solution, epoch, agent_idx):
        """Apply PSO-style magnitude scaling"""
        diversity = self._calculate_population_diversity()

        # PSO-style coefficients
        w = 0.9 - 0.5 * (epoch / self.epoch)  # Inertia weight decreases over time
        c1 = 2.0 * (1.0 + diversity)  # Cognitive component increases with diversity
        c2 = 2.0 * (2.0 - diversity)  # Social component decreases with diversity

        # Use predicted direction as "personal best" direction
        # and direction toward population mean as "global best"
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)
        global_direction = x_mean - current_solution

        # PSO-style velocity update using predicted direction
        velocity = (w * predicted_direction +
                   c1 * self.generator.random() * predicted_direction +
                   c2 * self.generator.random() * global_direction)

        # Scale velocity to reasonable magnitude
        velocity_magnitude = np.linalg.norm(velocity)
        bounds_range = np.linalg.norm(np.array(self.problem.ub) - np.array(self.problem.lb))
        max_velocity = 0.1 * bounds_range

        if velocity_magnitude > max_velocity:
            velocity = velocity * (max_velocity / velocity_magnitude)

        return current_solution + velocity

    def _apply_abc_style_magnitude(self, predicted_direction, current_solution, epoch, agent_idx):
        """Apply ABC-style magnitude scaling (single difference with random scaling)"""
        # ABC-style random scaling factor
        phi = self.generator.uniform(-1, 1)

        # Adaptive scaling based on epoch progress
        progress_ratio = epoch / self.epoch
        scale_factor = 1.0 - 0.5 * progress_ratio  # Reduce exploration over time

        # Apply ABC-style update: x_i + phi * (predicted_direction)
        mutant = current_solution + phi * scale_factor * predicted_direction

        return mutant

    def _apply_crossover(self, mutant_solution, current_solution):
        """Apply binomial crossover between mutant and current solution"""
        trial_solution = current_solution.copy()

        # Ensure at least one dimension is taken from mutant
        j_rand = self.generator.integers(0, self.problem.n_dims)

        for j in range(self.problem.n_dims):
            if self.generator.random() < self.crossover_rate or j == j_rand:
                trial_solution[j] = mutant_solution[j]

        return trial_solution

    def _apply_enhanced_magnitude_adjustment(self, predicted_direction, current_solution, epoch, agent_idx):
        """
        Enhanced magnitude adjustment using multiple population-based strategies.
        Adaptively selects the best strategy based on population state and progress.
        """
        if predicted_direction is None:
            return None

        diversity = self._calculate_population_diversity()
        progress_ratio = epoch / self.epoch

        # Strategy selection based on population diversity and progress
        if self.magnitude_strategy == 'simple':
            # Original simple magnitude scaling
            if not hasattr(self, 'base_magnitude'):
                self.base_magnitude = 0.1 * np.linalg.norm(np.array(self.problem.ub) - np.array(self.problem.lb))
            magnitude = self.base_magnitude * (0.95 ** epoch)
            mutant = current_solution + magnitude * predicted_direction

        elif self.magnitude_strategy == 'de_style':
            mutant = self._apply_de_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)

        elif self.magnitude_strategy == 'pso_style':
            mutant = self._apply_pso_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)

        elif self.magnitude_strategy == 'abc_style':
            mutant = self._apply_abc_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)

        elif self.magnitude_strategy == 'adaptive_multi':
            # Adaptive strategy selection based on population state
            if diversity > self.diversity_threshold and progress_ratio < 0.5:
                # High diversity, early stage: use DE-style for exploitation
                mutant = self._apply_de_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)
            elif diversity <= self.diversity_threshold and progress_ratio < 0.7:
                # Low diversity, mid stage: use PSO-style for balanced exploration
                mutant = self._apply_pso_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)
            else:
                # Late stage or very low diversity: use ABC-style for fine-tuning
                mutant = self._apply_abc_style_magnitude(predicted_direction, current_solution, epoch, agent_idx)
        else:
            # Fallback to simple strategy
            if not hasattr(self, 'base_magnitude'):
                self.base_magnitude = 0.1 * np.linalg.norm(np.array(self.problem.ub) - np.array(self.problem.lb))
            magnitude = self.base_magnitude * (0.95 ** epoch)
            mutant = current_solution + magnitude * predicted_direction

        # Apply crossover if enabled
        if self.crossover_rate > 0:
            trial_solution = self._apply_crossover(mutant, current_solution)
        else:
            trial_solution = mutant

        return trial_solution

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Collect training data: direction from worse solution to better solution.
        This creates consistent training signal regardless of minmax setting.
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
        direction = better_agent.solution - worse_agent.solution
        if np.linalg.norm(direction) > 1e-12:  # Only collect meaningful directions
            self.data.append({
                'start_pos': worse_agent.solution.copy(),
                'direction': direction.copy()
            })

        return better_agent.copy()

    def _predict_direction(self, current_position):
        """
        Use the trained neural network to predict a direction from the current position.
        Returns the predicted direction vector.
        """
        if not self.model_trained or self.dirnet is None:
            return None

        try:
            self.dirnet.eval()
            with torch.no_grad():
                # Normalize input using position bounds (same as training)
                pos_tensor = torch.tensor(current_position, dtype=torch.float32).to(self.device)
                pos_scale = ((torch.tensor(self.problem.ub, dtype=torch.float32) -
                            torch.tensor(self.problem.lb, dtype=torch.float32)) / 2.0).to(self.device)
                pos_shift = ((torch.tensor(self.problem.ub, dtype=torch.float32) +
                            torch.tensor(self.problem.lb, dtype=torch.float32)) / 2.0).to(self.device)
                norm_input = (pos_tensor - pos_shift) / pos_scale

                # Get prediction
                predicted_direction = self.dirnet(norm_input.unsqueeze(0)).squeeze(0)
                return predicted_direction.cpu().numpy()

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Direction prediction error: {e}")
            return None

    def evolve(self, epoch):
        """
        Enhanced evolve method with 50% probability neural network usage and online learning
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

        for idx in range(self.pop_size):
            current_solution = self.pop[idx].solution.copy()

            # Decide whether to use neural network prediction (50% probability when trained)
            use_prediction = (self.model_trained and
                            self.dirnet is not None and
                            self.generator.random() < self.prediction_usage_probability)

            if use_prediction:
                # Use neural network to predict direction
                predicted_direction = self._predict_direction(current_solution)

                if predicted_direction is not None:
                    # Apply enhanced magnitude adjustment with population-based strategies
                    pos_new = self._apply_enhanced_magnitude_adjustment(
                        predicted_direction, current_solution, epoch, idx)

                    if pos_new is None:
                        # Fallback to original PLO if enhanced adjustment fails
                        use_prediction = False
                else:
                    # Fallback to original PLO if prediction fails
                    use_prediction = False

            if not use_prediction:
                # Original PLO aurora-oval walk + Lévy flight
                a = self.generator.uniform() / 2 + 1
                V = np.exp((1 - a) / 100 * epoch)
                LS = V
                GS = (self.levy(self.problem.n_dims) *
                      (x_mean - current_solution +
                       (self.problem.lb + self.generator.uniform(0, 1, self.problem.n_dims) *
                        (self.problem.ub - self.problem.lb)) / 2))
                pos_new = (current_solution +
                           (w1 * LS + w2 * GS) *
                           self.generator.uniform(0, 1, self.problem.n_dims))

            # Optional collision (additional random perturbation)
            if self.generator.random() < 0.05 and self.generator.random() < E:
                j = self.generator.integers(0, self.problem.n_dims)
                pos_new[j] += np.sin(self.generator.random() * np.pi) * \
                              (current_solution[j] - self.pop[A[idx]].solution[j])

            # Clip to bounds and create new agent
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                # Collect data and update population
                self.pop[idx] = self._collect_direction_data(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            # Collect data during population update
            for i, (old_agent, new_agent) in enumerate(zip(self.pop, pop_new)):
                better_agent = self._collect_direction_data(new_agent, old_agent, self.problem.minmax)
                self.pop[i] = better_agent

        # Adaptive training frequency: train less often as algorithm progresses
        progress_ratio = epoch / self.epoch
        adaptive_train_every = max(self.train_every, int(self.train_every * (1 + 2 * progress_ratio)))

        if len(self.data) > 1024:
            self.data = self.data[-1024:]

        # Online learning: retrain when sufficient data is available
        if (epoch % adaptive_train_every == 0 and
            len(self.data) >= self.min_data_for_training):
            self._train_neural_network(epoch)

    def _train_neural_network(self, epoch):
        """
        Train the neural network with data augmentation, online learning, and comprehensive TensorBoard logging.
        Generates 100k+ samples through augmentation before training.
        """
        if len(self.data) < self.min_data_for_training:
            return

        # Increment training session counter
        writer = None

        try:
            # Step 2: Initialize TensorBoard writer for this training session
            if False and TENSORBOARD_AVAILABLE:
                try:
                    writer = SummaryWriter(log_dir='./logs/')
                except Exception as e:
                    print(f"Warning: Failed to initialize TensorBoard writer: {e}")
                    writer = None
            else:
                writer = None

            # Step 3: Create train/validation split
            train_size = int(0.8 * len(self.data))
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

            train_data = [self.data[i] for i in indices[:train_size]]
            val_data = [self.data[i] for i in indices[train_size:]]

            # Step 4: Create datasets and data loaders
            train_dataset = CustomDataset(train_data, self.problem.lb, self.problem.ub)
            dir_stats = train_dataset.get_direction_stats()
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub, pos_stats=dir_stats)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)

            # Step 5: Initialize network if needed
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims).to(self.device)

            # Step 6: Setup optimizer (simple Adam without weight decay)
            self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Step 7: Training loop with early stopping and comprehensive logging
            best_val_loss = float('inf')
            patience_counter = 0

            for e in range(self.n_grad_epochs):
                # Training phase
                self.dirnet.train()
                train_loss_total = 0.0
                train_batches = 0

                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    # Forward pass
                    predictions = self.dirnet(batch_x)
                    loss = self.criterion(predictions, batch_y)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.dirnet.parameters(), 1.0)
                    self.optimizer.step()

                    train_loss_total += loss.item()
                    train_batches += 1

                # Validation phase
                self.dirnet.eval()
                val_loss_total = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        predictions = self.dirnet(batch_x)
                        loss = self.criterion(predictions, batch_y)
                        val_loss_total += loss.item()
                        val_batches += 1

                # Calculate average losses
                avg_train_loss = train_loss_total / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                # Early stopping check
                if avg_val_loss < best_val_loss - self.min_loss_improvement:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        # Early stopping triggered
                        break

                # Log to TensorBoard
                if writer is not None:
                    try:
                        # Log losses using add_scalars for combined plot
                        writer.add_scalars(f'Loss/{self.problem.name}/', {
                            'train': avg_train_loss,
                            'val': avg_val_loss
                        }, self.global_step_counter)

                        # Log learning rate
                        writer.add_scalar(f'Training/{self.problem.name}/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step_counter)

                    except Exception as e:
                        print(f"Warning: Failed to log to TensorBoard: {e}")

                # Increment global step counter
                self.global_step_counter += 1

            # Mark model as trained
            self.model_trained = True

        except Exception as e:
            print(f"Error during neural network training: {e}")
            if self.logger:
                self.logger.error(f"Neural network training failed: {e}")

        finally:
            # Always close the TensorBoard writer
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:
                    print(f"Warning: Failed to close TensorBoard writer: {e}")