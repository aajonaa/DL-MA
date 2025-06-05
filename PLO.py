import numpy as np
from scipy.special import gamma
from optimizer import Optimizer

# Add PyTorch imports for KLM functionality
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

    def __init__(self, data, lb, ub):
        self.data_list = data
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

        # Position normalization parameters
        self.pos_scale = ((self.ub - self.lb) / 2.0).clone().detach()
        self.pos_shift = ((self.ub + self.lb) / 2.0).clone().detach()

        # Preprocess all samples once during initialization
        self.processed_samples = []
        for item in data:
            # Normalize position
            feature = torch.tensor(item['start_pos'], dtype=torch.float32)
            norm_feature = (feature - self.pos_shift) / self.pos_scale
            
            # Convert direction to unit vector
            direction = torch.tensor(item['direction'], dtype=torch.float32)
            norm = torch.norm(direction)
            unit_direction = direction / torch.clamp(norm, min=1e-8)
            
            self.processed_samples.append({
                'feature': norm_feature,
                'direction': unit_direction
            })

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        # Simply return preprocessed data
        sample = self.processed_samples[idx]
        return sample['feature'], sample['direction']


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred: predicted directions (batch_size, dim)
            target: ground truth unit vectors (batch_size, dim)
        Returns:
            loss: scalar
        """
        return 1 - F.cosine_similarity(pred, target, dim=1).mean()

    
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
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

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


class NDGPLO(OriginalPLO):
    """Neural Direction Guided PLO with simple classical operators - identical to DirPLO except for mutation operators"""

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=0.5,
                 min_data_for_training=256,  # Much lower for small problems
                 train_every=50,  # Back to more frequent training
                 n_grad_epochs=3,  # Reduced from 5 (fewer training epochs)
                 batch_size=32,  # Reasonable batch size
                 hidden_nodes=8,  # Reduced network size
                 learning_rate=1e-3,  # Increased learning rate for faster convergence
                 # Simple classical operator parameters (ENHANCED with BSA-inspired operator)
                 operator_strategy='bsa_inspired',  # 'adaptive_de_crossover', 'pso_guided_mutation', 'bsa_inspired', 'adaptive'
                 base_f_factor=0.5,
                 f_factor_range=(0.2, 0.8),
                 crossover_rate=0.7,
                 diversity_threshold=0.1,
                 **kwargs):
        super().__init__(epoch, pop_size, **kwargs)

        # Core parameters (identical to DirPLO)
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Enhanced classical operator parameters (ONLY DIFFERENCE from DirPLO)
        self.operator_strategy = operator_strategy
        self.base_f_factor = base_f_factor
        self.f_factor_range = f_factor_range
        self.crossover_rate = crossover_rate
        self.diversity_threshold = diversity_threshold

        # Advanced operator tracking - NOW 3 powerful operators
        self.operator_success_rates = {
            'adaptive_de_crossover': 0.33,  # DE-style with predicted direction guidance
            'pso_guided_mutation': 0.33,    # PSO-style with population guidance
            'bsa_inspired': 0.33             # BSA-style with historical population guidance
        }
        self.operator_usage_counts = {op: 0 for op in self.operator_success_rates.keys()}
        self.operator_success_counts = {op: 0 for op in self.operator_success_rates.keys()}
        self.last_operator_used = None

        # Historical population for BSA-inspired operator
        self.historical_population = []
        self.max_historical_size = max(10, pop_size // 2)  # Store up to half population size, minimum 10

        # Neural network components (identical to DirPLO)
        self.dirnet = None
        self.optimizer = None
        self.criterion = nn.MSELoss()  # Simple MSE loss for direction prediction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management (identical to DirPLO)
        self.data = []
        self.train_loader = None
        self.val_loader = None

        # Statistics for noise generation (identical to DirPLO)
        self.noise_std = None

        # Training session tracking for TensorBoard (identical to DirPLO)
        self.global_step_counter = 0

        # Performance optimization parameters (identical to DirPLO)
        self.early_stopping_patience = 3  # Stop training if no improvement
        self.min_loss_improvement = 1e-5  # Minimum improvement threshold
        self.last_training_loss = float('inf')
        self.no_improvement_count = 0

    @staticmethod
    def _cosine_loss(pred, tgt):
        """Cosine loss function (identical to DirPLO)"""
        pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
        tgt = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-12)
        return 1 - (pred * tgt).sum(dim=1).mean()

    @staticmethod
    def _mse_loss(pred, tgt):
        """Alternative MSE loss for direction prediction (identical to DirPLO)"""
        return torch.nn.functional.mse_loss(pred, tgt)

    @staticmethod
    def _combined_loss(pred, tgt, alpha=0.7):
        """Combined cosine and MSE loss for better stability (identical to DirPLO)"""
        cosine_loss = NDGPLO._cosine_loss(pred, tgt)
        mse_loss = NDGPLO._mse_loss(pred, tgt)
        return alpha * cosine_loss + (1 - alpha) * mse_loss

    def _calculate_population_diversity(self):
        """Calculate population diversity for adaptive magnitude scaling (identical to DirPLO)"""
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

    def _get_adaptive_f_factor(self, epoch):
        """Adaptive F factor - smaller values in later stages for exploitation"""
        progress_ratio = epoch / self.epoch

        if progress_ratio > 0.7:
            # Late stage: bias toward smaller F values for fine-tuning
            # Use lower 30% of the F range more frequently
            range_size = self.f_factor_range[1] - self.f_factor_range[0]
            if self.generator.random() < 0.7:  # 70% chance of small F
                f_factor = self.generator.uniform(self.f_factor_range[0],
                                                self.f_factor_range[0] + 0.3 * range_size)
            else:  # 30% chance of normal F
                f_factor = self.generator.uniform(self.f_factor_range[0], self.f_factor_range[1])
        else:
            # Early-mid stage: normal random F factor
            f_factor = self.generator.uniform(self.f_factor_range[0], self.f_factor_range[1])

        return f_factor

    def _apply_adaptive_de_crossover_operator(self, predicted_direction, current_solution, epoch):
        """
        Simple DE-style operator: Exploration-focused with predicted direction guidance
        """
        progress_ratio = epoch / self.epoch

        # Adaptive F factor - starts high for exploration, decreases for exploitation
        F = 0.8 * (1.0 - 0.6 * progress_ratio)  # 0.8 -> 0.32

        # Get best solution
        best_agent = min(self.pop, key=lambda x: x.target.fitness if self.problem.minmax == "min"
                         else lambda x: -x.target.fitness)
        best_pos = best_agent.solution

        # Simple DE-style with predicted direction as primary mutation vector
        new_solution = best_pos + F * predicted_direction

        return self.correct_solution(new_solution)

    def _apply_pso_guided_mutation_operator(self, predicted_direction, current_solution, epoch):
        """
        Simple PSO-style operator: Exploitation-focused with population guidance
        """
        progress_ratio = epoch / self.epoch

        # Adaptive scaling factors
        alpha = 0.6 + 0.3 * progress_ratio  # Increases over time (0.6 -> 0.9)
        beta = 0.4 - 0.2 * progress_ratio   # Decreases over time (0.4 -> 0.2)

        # Population mean
        X_mean = np.mean([agent.solution for agent in self.pop], axis=0)

        # Simple PSO-style with predicted direction and population guidance
        new_solution = (current_solution +
                       alpha * predicted_direction +
                       beta * (X_mean - current_solution))

        return self.correct_solution(new_solution)

    def _apply_bsa_inspired_operator(self, predicted_direction, current_solution, epoch):
        """
        BSA-inspired operator: Balanced exploration-exploitation with historical population guidance
        Uses BSA's recombination mechanism with predicted_direction as guidance
        """
        progress_ratio = epoch / self.epoch

        # Adaptive F factor - balanced scaling throughout the process
        F = 0.5 + 0.3 * np.sin(np.pi * progress_ratio)  # 0.5 -> 0.8 -> 0.5 (sinusoidal)

        # Get historical solution (if available) or use population mean as fallback
        if len(self.historical_population) > 0:
            # Select random historical solution
            historical_idx = self.generator.integers(0, len(self.historical_population))
            historical_solution = self.historical_population[historical_idx].solution
        else:
            # Fallback to population mean if no historical data
            historical_solution = np.mean([agent.solution for agent in self.pop], axis=0)

        # Adaptive dimension rate (DIM_RATE) - controls how many dimensions to update
        dim_rate = 0.3 + 0.4 * (1.0 - progress_ratio)  # 0.7 -> 0.3 (decreases over time)

        # Create dimension-wise mask (map) for selective updating
        mask = self.generator.random(self.problem.n_dims) < dim_rate

        # Ensure at least one dimension is updated
        if not np.any(mask):
            mask[self.generator.integers(0, self.problem.n_dims)] = True

        # BSA-inspired recombination: current + (mask * F) * (historical - current)
        # Note: (historical_solution - current_solution) serves as our predicted_direction equivalent
        historical_direction = historical_solution - current_solution

        # Apply BSA recombination with mask
        new_solution = current_solution.copy()
        new_solution[mask] = current_solution[mask] + F * historical_direction[mask]

        # Optional: blend with predicted_direction for enhanced guidance
        blend_factor = 0.3 * progress_ratio  # Increases over time (0.0 -> 0.3)
        if blend_factor > 0:
            new_solution[mask] = ((1 - blend_factor) * new_solution[mask] +
                                 blend_factor * (current_solution[mask] + F * predicted_direction[mask]))

        return self.correct_solution(new_solution)

    def _update_historical_population(self, current_population):
        """
        Update historical population with good solutions from current population
        """
        # Sort current population by fitness
        sorted_pop = sorted(current_population, key=lambda x: x.target.fitness,
                          reverse=(self.problem.minmax == "max"))

        # Add top solutions to historical population
        num_to_add = min(3, len(sorted_pop) // 4)  # Add top 25% or 3 solutions, whichever is smaller
        for i in range(num_to_add):
            if len(self.historical_population) < self.max_historical_size:
                self.historical_population.append(sorted_pop[i].copy())
            else:
                # Replace worst historical solution if current is better
                worst_historical_idx = 0
                worst_fitness = self.historical_population[0].target.fitness
                for j in range(1, len(self.historical_population)):
                    current_fitness = self.historical_population[j].target.fitness
                    if ((self.problem.minmax == "min" and current_fitness > worst_fitness) or
                        (self.problem.minmax == "max" and current_fitness < worst_fitness)):
                        worst_fitness = current_fitness
                        worst_historical_idx = j

                # Replace if current solution is better than worst historical
                current_fitness = sorted_pop[i].target.fitness
                if ((self.problem.minmax == "min" and current_fitness < worst_fitness) or
                    (self.problem.minmax == "max" and current_fitness > worst_fitness)):
                    self.historical_population[worst_historical_idx] = sorted_pop[i].copy()

    def _apply_crossover(self, mutant_solution, current_solution):
        """Apply binomial crossover between mutant and current solution (identical to DirPLO)"""
        trial_solution = current_solution.copy()

        # Ensure at least one dimension is taken from mutant
        j_rand = self.generator.integers(0, self.problem.n_dims)

        for j in range(self.problem.n_dims):
            if self.generator.random() < self.crossover_rate or j == j_rand:
                trial_solution[j] = mutant_solution[j]

        return trial_solution

    def _update_operator_success_rate(self, operator_name, success):
        """Enhanced success rate tracking with higher weight for recent performance"""
        if operator_name in self.operator_success_rates:
            self.operator_usage_counts[operator_name] += 1
            if success:
                self.operator_success_counts[operator_name] += 1

            # Enhanced exponential moving average with higher alpha for faster adaptation
            current_rate = self.operator_success_counts[operator_name] / max(1, self.operator_usage_counts[operator_name])
            alpha = 0.3  # Increased from 0.1 to give more weight to recent performance
            self.operator_success_rates[operator_name] = (
                alpha * current_rate + (1 - alpha) * self.operator_success_rates[operator_name]
            )

    def _select_operator_adaptively(self, epoch):
        """Advanced operator selection with research-backed adaptive strategy - NOW 3 powerful operators"""
        if self.operator_strategy == 'adaptive':
            progress_ratio = epoch / self.epoch
            de_success = self.operator_success_rates['adaptive_de_crossover']
            pso_success = self.operator_success_rates['pso_guided_mutation']
            bsa_success = self.operator_success_rates['bsa_inspired']

            # Research-backed adaptive selection strategy for 3 operators
            if progress_ratio < 0.25:
                # Early stage: favor PSO for exploration, some BSA for diversity
                de_prob, pso_prob, bsa_prob = 0.2, 0.5, 0.3
            elif progress_ratio < 0.5:
                # Early-mid stage: balanced with slight PSO bias
                de_prob, pso_prob, bsa_prob = 0.3, 0.4, 0.3
            elif progress_ratio < 0.75:
                # Mid-late stage: favor DE for exploitation, BSA for guidance
                de_prob, pso_prob, bsa_prob = 0.5, 0.2, 0.3
            else:
                # Late stage: strong DE bias, BSA for fine-tuning
                de_prob, pso_prob, bsa_prob = 0.6, 0.15, 0.25

            # Success rate adaptation with sensitivity for 3 operators
            total_success = de_success + pso_success + bsa_success
            if total_success > 0:
                # Normalize success rates
                de_norm = de_success / total_success
                pso_norm = pso_success / total_success
                bsa_norm = bsa_success / total_success

                # Adaptive boost based on relative performance
                boost_factor = 0.15
                if de_norm > 0.4:  # DE performing well
                    de_prob = min(0.8, de_prob + boost_factor)
                    pso_prob = max(0.1, pso_prob - boost_factor/2)
                    bsa_prob = max(0.1, bsa_prob - boost_factor/2)
                elif pso_norm > 0.4:  # PSO performing well
                    pso_prob = min(0.8, pso_prob + boost_factor)
                    de_prob = max(0.1, de_prob - boost_factor/2)
                    bsa_prob = max(0.1, bsa_prob - boost_factor/2)
                elif bsa_norm > 0.4:  # BSA performing well
                    bsa_prob = min(0.8, bsa_prob + boost_factor)
                    de_prob = max(0.1, de_prob - boost_factor/2)
                    pso_prob = max(0.1, pso_prob - boost_factor/2)

            # Normalize probabilities to sum to 1
            total_prob = de_prob + pso_prob + bsa_prob
            de_prob /= total_prob
            pso_prob /= total_prob
            bsa_prob /= total_prob

            # Select operator using cumulative probabilities
            rand_val = self.generator.random()
            if rand_val < de_prob:
                return 'adaptive_de_crossover'
            elif rand_val < de_prob + pso_prob:
                return 'pso_guided_mutation'
            else:
                return 'bsa_inspired'
        else:
            # Return the specified strategy (fallback to DE if not recognized)
            if self.operator_strategy in self.operator_success_rates:
                return self.operator_strategy
            else:
                return 'adaptive_de_crossover'

    def _apply_advanced_research_operators(self, predicted_direction, current_solution, epoch):
        """
        Advanced research-backed operators - NOW 3 powerful operators with sophisticated mechanisms.
        Uses predicted_direction as mutation vector, crossover guidance, and masking intelligence.
        This replaces DirPLO's complex _apply_enhanced_magnitude_adjustment method.
        """
        if predicted_direction is None:
            return None

        # Select operator using advanced adaptive mechanism
        selected_operator = self._select_operator_adaptively(epoch)

        # Apply the selected advanced operator (now 3 options)
        if selected_operator == 'adaptive_de_crossover':
            trial_solution = self._apply_adaptive_de_crossover_operator(predicted_direction, current_solution, epoch)
        elif selected_operator == 'pso_guided_mutation':
            trial_solution = self._apply_pso_guided_mutation_operator(predicted_direction, current_solution, epoch)
        elif selected_operator == 'bsa_inspired':
            trial_solution = self._apply_bsa_inspired_operator(predicted_direction, current_solution, epoch)
        else:
            # Fallback to adaptive DE crossover
            trial_solution = self._apply_adaptive_de_crossover_operator(predicted_direction, current_solution, epoch)
            selected_operator = 'adaptive_de_crossover'

        # Store the operator used for success rate tracking
        self.last_operator_used = selected_operator

        # Note: Advanced operators already include sophisticated crossover mechanisms
        # No additional crossover needed

        return trial_solution

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Enhanced data collection with operator success rate tracking (based on DirPLO).
        This creates consistent training signal regardless of minmax setting.
        """
        # Determine which agent is better based on minmax
        if minmax == "min":
            # For minimization, lower fitness is better
            if agent_x.target.fitness < agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
                success = True
            else:
                better_agent = agent_y
                worse_agent = agent_x
                success = False
        else:  # maxmax == "max"
            # For maximization, higher fitness is better
            if agent_x.target.fitness > agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
                success = True
            else:
                better_agent = agent_y
                worse_agent = agent_x
                success = False

        # Update operator success rate if we have a record of the last operator used
        if self.last_operator_used is not None:
            self._update_operator_success_rate(self.last_operator_used, success)
            self.last_operator_used = None  # Reset for next iteration

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
        Use the trained neural network to predict a direction from the current position (identical to DirPLO).
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

    def _train_neural_network(self, epoch):
        """
        Train the neural network with data augmentation, online learning, and comprehensive TensorBoard logging (identical to DirPLO).
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
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

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

    def evolve(self, epoch):
        """
        Enhanced evolve method with 50% probability neural network usage and online learning (identical to DirPLO except for mutation operators)
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
                    # Apply advanced research-backed operators instead of DirPLO's complex magnitude adjustment
                    pos_new = self._apply_advanced_research_operators(
                        predicted_direction, current_solution, epoch)

                    if pos_new is None:
                        # Fallback to original PLO if advanced operators fail
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

        # Update historical population for BSA-inspired operator
        # Do this every few epochs to maintain good historical solutions
        if epoch % 5 == 0:  # Update every 5 epochs
            self._update_historical_population(self.pop)

        # Adaptive training frequency: train less often as algorithm progresses
        progress_ratio = epoch / self.epoch
        adaptive_train_every = max(self.train_every, int(self.train_every * (1 + 2 * progress_ratio)))

        if len(self.data) > 1024:
            self.data = self.data[-1024:]

        # Online learning: retrain when sufficient data is available
        if (epoch % adaptive_train_every == 0 and
            len(self.data) >= self.min_data_for_training):
            self._train_neural_network(epoch)


class SimpleBSANDGPLO(OriginalPLO):
    """
    CORRECTED PURE BSA Algorithm Implementation for baseline performance evaluation.
    Runs 100% BSA algorithm using the CORRECT 5-step BSA process with proper formulas.
    Fixed: Selection-I logic, Mutation formula (Mutant = oldP + F*(P1-P2)), Crossover mapping.
    No PLO components - pure BSA to establish BSA's true optimization capabilities.
    """

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=0.7,  # More aggressive neural network usage
                 min_data_for_training=256,
                 train_every=50,
                 n_grad_epochs=3,
                 batch_size=32,
                 hidden_nodes=8,
                 learning_rate=1e-3,
                 # BSA-specific parameters
                 base_f_factor=0.5,
                 **kwargs):
        super().__init__(epoch, pop_size, **kwargs)

        # Core parameters
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.base_f_factor = base_f_factor

        # Neural network components
        self.dirnet = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management
        self.data = []
        self.train_loader = None
        self.val_loader = None

        # Historical population for BSA operator
        self.historical_population = []
        self.max_historical_size = max(10, pop_size // 2)
        self.historical_initialized = False  # Track if historical population has been initialized

        # Training optimization parameters
        self.early_stopping_patience = 3
        self.min_loss_improvement = 1e-5
        self.global_step_counter = 0

    @staticmethod
    def _mse_loss(pred, tgt):
        """Simple MSE loss for direction prediction"""
        return torch.mean((pred - tgt) ** 2)

    @staticmethod
    def _cosine_loss(pred, tgt):
        """Cosine similarity loss for direction alignment"""
        pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1)
        tgt_norm = torch.nn.functional.normalize(tgt, p=2, dim=1)
        cosine_sim = torch.sum(pred_norm * tgt_norm, dim=1)
        return torch.mean(1 - cosine_sim)

    @staticmethod
    def _combined_loss(pred, tgt, alpha=0.7):
        """Combined cosine and MSE loss for better stability"""
        cosine_loss = SimpleBSANDGPLO._cosine_loss(pred, tgt)
        mse_loss = SimpleBSANDGPLO._mse_loss(pred, tgt)
        return alpha * cosine_loss + (1 - alpha) * mse_loss

    def _apply_pure_bsa_algorithm(self, epoch):
        """
        CORRECTED BSA Algorithm Implementation: Standard 5-step BSA algorithm
        Based on Civicioglu 2013 and the reference implementation in the codebase
        Steps: Selection-I, Mutation, Crossover, Selection-II, Update
        """
        # Initialize historical population if not done
        if not hasattr(self, 'historical_population') or len(self.historical_population) == 0:
            self.historical_population = [agent.copy() for agent in self.pop]

        # === STEP 1: SELECTION-I ===
        # BSA Selection-I: Decide whether to update historical population
        a = self.generator.random()
        b = self.generator.random()

        if a < b:
            # Update historical population with current population
            self.historical_population = [agent.copy() for agent in self.pop]

        # Permute (shuffle) the historical population - CRITICAL BSA STEP
        permuted_indices = self.generator.permutation(len(self.historical_population))
        self.historical_population = [self.historical_population[i] for i in permuted_indices]

        # === STEP 2: MUTATION ===
        # BSA Mutation: Generate mutant population
        # CORRECTED Formula: Mutant = oldP + F * (P1 - P2) where P1, P2 are random individuals
        mutant_population = []

        # Calculate F factor - BSA uses different range than DE
        F = 3 * self.generator.random()  # BSA standard: F ∈ [0, 3]

        for i, agent in enumerate(self.pop):
            # Select random individuals P1 and P2 from current population
            random_indices = self.generator.choice(len(self.pop), size=2, replace=False)
            P1 = self.pop[random_indices[0]].solution
            P2 = self.pop[random_indices[1]].solution

            # Get corresponding historical individual (oldP)
            oldP = self.historical_population[i].solution

            # CORRECT BSA Mutation Formula: Mutant = oldP + F * (P1 - P2)
            mutant_solution = oldP + F * (P1 - P2)
            mutant_solution = self.correct_solution(mutant_solution)

            # Create mutant agent
            mutant_agent = self.generate_empty_agent(mutant_solution)
            mutant_agent.target = self.get_target(mutant_solution)
            mutant_population.append(mutant_agent)

        # === STEP 3: CROSSOVER ===
        # BSA Crossover: Create trial population using mapping strategy
        trial_population = []

        for i, (current_agent, mutant_agent) in enumerate(zip(self.pop, mutant_population)):
            # Initialize mapping matrix (all dimensions selected initially)
            mapping = np.ones(self.problem.n_dims, dtype=bool)

            # BSA Crossover Strategy 1: Random dimension deselection
            if self.generator.random() < self.generator.random():
                # Randomly set some positions to False based on mix_rate
                for j in range(self.problem.n_dims):
                    if self.generator.random() < 1.0:  # mix_rate = 1.0 in standard BSA
                        mapping[j] = False

            # BSA Crossover Strategy 2: Ensure at least one dimension is selected
            if not np.any(mapping):
                # If all positions are False, randomly select one position
                random_pos = self.generator.integers(0, self.problem.n_dims)
                mapping[random_pos] = True

            # Generate trial individual: inherit from mutant where mapping is True
            trial_solution = current_agent.solution.copy()
            trial_solution[mapping] = mutant_agent.solution[mapping]
            trial_solution = self.correct_solution(trial_solution)

            # Create trial agent
            trial_agent = self.generate_empty_agent(trial_solution)
            trial_agent.target = self.get_target(trial_solution)
            trial_population.append(trial_agent)

        # === STEP 5: SELECTION-II ===
        # BSA Selection-II: Select between current and trial populations
        new_population = []
        for current_agent, trial_agent in zip(self.pop, trial_population):
            # Select better individual using proper agent comparison
            if trial_agent.is_better_than(current_agent, self.problem.minmax):
                new_population.append(trial_agent)
            else:
                new_population.append(current_agent)

        # === STEP 6: FITNESS EVALUATION & UPDATE ===
        # Update population with new individuals
        self.pop = new_population

        # Update global best
        for agent in self.pop:
            if agent.is_better_than(self.g_best, self.problem.minmax):
                self.g_best = agent.copy()

        return True  # BSA algorithm completed successfully

    def _update_historical_population(self, current_population):
        """
        Standard BSA historical population management.
        Simple and conservative approach aligned with original BSA algorithm.
        """
        # Initialize historical population on first call
        if not self.historical_initialized:
            self.historical_population = [agent.copy() for agent in current_population]
            self.historical_initialized = True
            return

        # In standard BSA, historical population is updated with some probability
        update_probability = 0.5  # BSA standard update probability

        if self.generator.random() < update_probability:
            # Replace entire historical population with current population
            self.historical_population = [agent.copy() for agent in current_population]

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Simple data collection for neural network training.
        Collects direction data from worse to better solutions.
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
        Returns the predicted direction vector for BSA operator.
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

    def _train_neural_network(self, epoch):
        """
        Train the neural network with collected direction data.
        Simple and efficient training for BSA operator guidance.
        """
        if len(self.data) < self.min_data_for_training:
            return

        try:
            # Create train/validation split
            train_size = int(0.8 * len(self.data))
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

            train_data = [self.data[i] for i in indices[:train_size]]
            val_data = [self.data[i] for i in indices[train_size:]]

            # Create datasets and data loaders
            train_dataset = CustomDataset(train_data, self.problem.lb, self.problem.ub)
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)

            # Initialize network if needed
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims).to(self.device)

            # Setup optimizer
            self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Training loop with early stopping
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
                avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                # Early stopping check
                if avg_val_loss < best_val_loss - self.min_loss_improvement:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

                self.global_step_counter += 1

            # Mark model as trained
            self.model_trained = True

        except Exception as e:
            print(f"Error during neural network training: {e}")
            if self.logger:
                self.logger.error(f"Neural network training failed: {e}")

    def evolve(self, epoch):
        """
        PURE BSA evolve method for baseline performance evaluation.
        Uses 100% BSA algorithm - no PLO components or fallback mechanisms for solution updates.
        Establishes BSA's core optimization capabilities as a standalone algorithm.
        """
        # PURE BSA ONLY - Apply complete BSA algorithm for entire population
        self._apply_pure_bsa_algorithm(epoch)

        # Explicitly ensure no other updates occur outside BSA operators
        # Neural network training and data collection are disabled for solution updates
        # All solution updates are handled exclusively by BSA in _apply_pure_bsa_algorithm


# REMOVED: Conflicting BacktrackingSearchOptimization class
# The correct BSA implementation is in SimpleBSANDGPLO class above
# Use SimpleBSANDGPLO for all BSA experiments

# Alias for backward compatibility with experiment code
# This ensures existing experiment code that references BacktrackingSearchOptimization will work
BacktrackingSearchOptimization = SimpleBSANDGPLO


class NDGPLO_BSA(SimpleBSANDGPLO):
    """
    Neural Direction Guided BSA Algorithm - Enhanced BSA with predicted_direction integration.
    Combines the corrected BSA algorithm with intelligent neural network direction prediction.

    Key Features:
    - Enhanced BSA mutation: Blends BSA formula with predicted_direction
    - Smart crossover mapping: Prioritizes dimensions indicated by predicted_direction
    - Neural-guided historical selection: Uses predictions to improve BSA Selection-I
    - Maintains BSA algorithmic integrity while adding neural intelligence
    """

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=0.7,  # Probability of using neural predictions
                 min_data_for_training=256,
                 train_every=50,
                 n_grad_epochs=3,
                 batch_size=32,
                 hidden_nodes=8,
                 learning_rate=1e-3,
                 # Neural-BSA integration parameters
                 neural_blend_factor=0.3,  # How much to blend predicted_direction with BSA
                 dimension_priority_threshold=0.5,  # Threshold for dimension prioritization
                 adaptive_integration=True,  # Whether to adapt integration over time
                 **kwargs):

        # Initialize parent SimpleBSANDGPLO
        super().__init__(epoch, pop_size, **kwargs)

        # Neural network parameters (from DirPLO)
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Neural-BSA integration parameters
        self.neural_blend_factor = neural_blend_factor
        self.dimension_priority_threshold = dimension_priority_threshold
        self.adaptive_integration = adaptive_integration

        # Neural network components (from DirPLO)
        self.dirnet = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management for neural network training
        self.data = []

        # Training optimization parameters
        self.early_stopping_patience = 3
        self.min_loss_improvement = 1e-5
        self.global_step_counter = 0

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Collect direction data for neural network training.
        Collects direction from worse to better solutions for learning.
        """
        # Determine which agent is better based on minmax
        if minmax == "min":
            if agent_x.target.fitness < agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x
        else:  # maxmax == "max"
            if agent_x.target.fitness > agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x

        # Collect direction data: from worse position toward better position
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
        Returns the predicted direction vector for BSA enhancement.
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

    def _train_neural_network(self, epoch):
        """
        Train the neural network with collected direction data.
        Simple and efficient training for BSA enhancement.
        """
        if len(self.data) < self.min_data_for_training:
            return

        try:
            # Create train/validation split
            train_size = int(0.8 * len(self.data))
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

            train_data = [self.data[i] for i in indices[:train_size]]
            val_data = [self.data[i] for i in indices[train_size:]]

            # Create datasets and data loaders
            train_dataset = CustomDataset(train_data, self.problem.lb, self.problem.ub)
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)

            # Initialize network if needed
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims).to(self.device)

            # Setup optimizer
            self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Training loop with early stopping
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
                avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                # Early stopping check
                if avg_val_loss < best_val_loss - self.min_loss_improvement:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

                self.global_step_counter += 1

            # Mark model as trained
            self.model_trained = True

        except Exception as e:
            print(f"Error during neural network training: {e}")
            if self.logger:
                self.logger.error(f"Neural network training failed: {e}")

    def _apply_neural_guided_bsa_algorithm(self, epoch):
        """
        Neural-Guided BSA Algorithm: Enhanced BSA with predicted_direction integration.
        Maintains BSA algorithmic integrity while adding intelligent neural guidance.

        Steps: Selection-I, Neural-Enhanced Mutation, Neural-Guided Crossover, Selection-II
        """
        progress_ratio = epoch / self.epoch

        # Initialize historical population if not done
        if not hasattr(self, 'historical_population') or len(self.historical_population) == 0:
            self.historical_population = [agent.copy() for agent in self.pop]

        # === STEP 1: SELECTION-I (Standard BSA) ===
        # BSA Selection-I: Decide whether to update historical population
        a = self.generator.random()
        b = self.generator.random()

        if a < b:
            # Update historical population with current population
            self.historical_population = [agent.copy() for agent in self.pop]

        # Permute (shuffle) the historical population - CRITICAL BSA STEP
        permuted_indices = self.generator.permutation(len(self.historical_population))
        self.historical_population = [self.historical_population[i] for i in permuted_indices]

        # === STEP 2: NEURAL-ENHANCED MUTATION ===
        # Enhanced BSA Mutation: Blend BSA formula with predicted_direction
        mutant_population = []

        # Calculate F factor - BSA standard range
        F = 3 * self.generator.random()  # BSA standard: F ∈ [0, 3]

        for i, agent in enumerate(self.pop):
            # Standard BSA mutation components
            random_indices = self.generator.choice(len(self.pop), size=2, replace=False)
            P1 = self.pop[random_indices[0]].solution
            P2 = self.pop[random_indices[1]].solution
            oldP = self.historical_population[i].solution

            # Standard BSA mutation: Mutant = oldP + F * (P1 - P2)
            bsa_mutant = oldP + F * (P1 - P2)

            # Neural enhancement: Try to get predicted direction
            use_neural_guidance = (self.model_trained and
                                 self.dirnet is not None and
                                 self.generator.random() < self.prediction_usage_probability)

            if use_neural_guidance:
                predicted_direction = self._predict_direction(agent.solution)

                if predicted_direction is not None:
                    # ENHANCED BSA MUTATION: Blend BSA with predicted_direction
                    mutant_solution = self._enhanced_bsa_mutation(
                        agent.solution, bsa_mutant, predicted_direction, progress_ratio)
                else:
                    # Fallback to standard BSA mutation
                    mutant_solution = bsa_mutant
            else:
                # Use standard BSA mutation
                mutant_solution = bsa_mutant

            # Apply bounds correction
            mutant_solution = self.correct_solution(mutant_solution)

            # Create mutant agent
            mutant_agent = self.generate_empty_agent(mutant_solution)
            mutant_agent.target = self.get_target(mutant_solution)
            mutant_population.append(mutant_agent)

        # === STEP 3: NEURAL-GUIDED CROSSOVER ===
        # Enhanced BSA Crossover: Use predicted_direction to guide dimension selection
        trial_population = []

        for i, (current_agent, mutant_agent) in enumerate(zip(self.pop, mutant_population)):
            # Try to get predicted direction for crossover guidance
            use_neural_crossover = (self.model_trained and
                                  self.dirnet is not None and
                                  self.generator.random() < self.prediction_usage_probability)

            if use_neural_crossover:
                predicted_direction = self._predict_direction(current_agent.solution)

                if predicted_direction is not None:
                    # NEURAL-GUIDED CROSSOVER: Prioritize important dimensions
                    mapping = self._neural_guided_crossover_mapping(
                        predicted_direction, progress_ratio)
                else:
                    # Fallback to standard BSA crossover
                    mapping = self._standard_bsa_crossover_mapping()
            else:
                # Use standard BSA crossover
                mapping = self._standard_bsa_crossover_mapping()

            # Generate trial individual: inherit from mutant where mapping is True
            trial_solution = current_agent.solution.copy()
            trial_solution[mapping] = mutant_agent.solution[mapping]
            trial_solution = self.correct_solution(trial_solution)

            # Create trial agent
            trial_agent = self.generate_empty_agent(trial_solution)
            trial_agent.target = self.get_target(trial_solution)
            trial_population.append(trial_agent)

        # === STEP 4: SELECTION-II (Standard BSA) ===
        # BSA Selection-II: Select between current and trial populations
        new_population = []
        for current_agent, trial_agent in zip(self.pop, trial_population):
            # Collect direction data for neural network training
            better_agent = self._collect_direction_data(current_agent, trial_agent, self.problem.minmax)

            # Select better individual using proper agent comparison
            if trial_agent.is_better_than(current_agent, self.problem.minmax):
                new_population.append(trial_agent)
            else:
                new_population.append(current_agent)

        # === STEP 5: UPDATE POPULATION ===
        # Update population with new individuals
        self.pop = new_population

        # Update global best
        for agent in self.pop:
            if agent.is_better_than(self.g_best, self.problem.minmax):
                self.g_best = agent.copy()

        return True  # Neural-guided BSA algorithm completed successfully

    def _enhanced_bsa_mutation(self, current_solution, bsa_mutant, predicted_direction, progress_ratio):
        """
        Enhanced BSA Mutation: Intelligently blend BSA mutation with predicted_direction.

        Args:
            current_solution: Current agent position
            bsa_mutant: Standard BSA mutation result
            predicted_direction: Neural network predicted direction
            progress_ratio: Algorithm progress (0 to 1)

        Returns:
            Enhanced mutant solution
        """
        # Adaptive blending factor based on progress and configuration
        if self.adaptive_integration:
            # Start with more BSA, gradually increase neural influence
            blend_factor = self.neural_blend_factor * progress_ratio
        else:
            # Fixed blending factor
            blend_factor = self.neural_blend_factor

        # Ensure blend factor is in valid range
        blend_factor = np.clip(blend_factor, 0.0, 0.8)  # Max 80% neural influence

        # Method 1: Weighted combination of BSA and neural directions
        # BSA direction: bsa_mutant - current_solution
        bsa_direction = bsa_mutant - current_solution

        # Combine directions with adaptive weighting
        combined_direction = ((1 - blend_factor) * bsa_direction +
                            blend_factor * predicted_direction)

        # Generate enhanced mutant
        enhanced_mutant = current_solution + combined_direction

        return enhanced_mutant

    def _neural_guided_crossover_mapping(self, predicted_direction, progress_ratio):
        """
        Neural-Guided Crossover Mapping: Use predicted_direction to prioritize dimensions.

        Args:
            predicted_direction: Neural network predicted direction
            progress_ratio: Algorithm progress (0 to 1)

        Returns:
            Boolean mapping array for crossover
        """
        # Calculate dimension importance based on predicted_direction magnitude
        direction_magnitude = np.abs(predicted_direction)

        # Normalize to [0, 1] range
        if np.max(direction_magnitude) > 1e-12:
            normalized_importance = direction_magnitude / np.max(direction_magnitude)
        else:
            # Fallback to uniform importance if direction is too small
            normalized_importance = np.ones(len(predicted_direction)) * 0.5

        # Adaptive threshold based on progress
        if self.adaptive_integration:
            # Start conservative, become more selective over time
            threshold = self.dimension_priority_threshold * (1 + progress_ratio)
        else:
            threshold = self.dimension_priority_threshold

        # Create mapping based on importance threshold
        mapping = normalized_importance > threshold

        # Ensure at least one dimension is selected (BSA requirement)
        if not np.any(mapping):
            # Select the most important dimension
            most_important_dim = np.argmax(normalized_importance)
            mapping[most_important_dim] = True

        # Ensure not all dimensions are selected (maintain some BSA diversity)
        if np.all(mapping) and len(mapping) > 1:
            # Randomly deselect some less important dimensions
            less_important_dims = np.where(normalized_importance < np.median(normalized_importance))[0]
            if len(less_important_dims) > 0:
                deselect_count = max(1, len(less_important_dims) // 3)
                deselect_dims = self.generator.choice(less_important_dims, size=deselect_count, replace=False)
                mapping[deselect_dims] = False

        return mapping

    def _standard_bsa_crossover_mapping(self):
        """
        Standard BSA Crossover Mapping: Original BSA crossover strategy.

        Returns:
            Boolean mapping array for crossover
        """
        # Initialize mapping matrix (all dimensions selected initially)
        mapping = np.ones(self.problem.n_dims, dtype=bool)

        # BSA Crossover Strategy 1: Random dimension deselection
        if self.generator.random() < self.generator.random():
            # Randomly set some positions to False based on mix_rate
            for j in range(self.problem.n_dims):
                if self.generator.random() < 1.0:  # mix_rate = 1.0 in standard BSA
                    mapping[j] = False

        # BSA Crossover Strategy 2: Ensure at least one dimension is selected
        if not np.any(mapping):
            # If all positions are False, randomly select one position
            random_pos = self.generator.integers(0, self.problem.n_dims)
            mapping[random_pos] = True

        return mapping

    def evolve(self, epoch):
        """
        NDGPLO_BSA evolve method: Neural-guided BSA with intelligent direction prediction.

        Algorithm Flow:
        1. Apply neural-guided BSA algorithm (enhanced BSA with predicted_direction)
        2. Train neural network periodically with collected direction data
        3. Fallback to pure BSA when neural predictions are unavailable
        """
        # Apply neural-guided BSA algorithm
        self._apply_neural_guided_bsa_algorithm(epoch)

        # Periodic neural network training
        if (epoch % self.train_every == 0 and
            len(self.data) >= self.min_data_for_training):
            self._train_neural_network(epoch)

        # Limit data size to prevent memory issues
        if len(self.data) > 1024:
            self.data = self.data[-1024:]


# Only using the mutant assisted by the network
class NDGPLO(OriginalPLO):
    """
    IMPROVED Neural Direction Guided PLO - Conservative Neural Integration

    Based on successful patterns from NDGPLO_BSA:
    - Conservative neural integration with gradual progression
    - Simple, robust blending strategy inspired by NDGPLO_BSA success
    - Focus on enhancing PLO's core movement rather than all components
    - Reliable fallback to pure PLO when neural predictions unavailable

    Key Improvements:
    - Single conservative enhancement factor (like NDGPLO_BSA)
    - Progressive neural influence starting from 0%
    - Simplified integration without multi-component interference
    - Robust fallback mechanisms
    """

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=1.0,  # Probability of using neural predictions
                 min_data_for_training=256,
                 train_every=50,
                 n_grad_epochs=3,
                 batch_size=32,
                 hidden_nodes=8,
                 learning_rate=1e-3,
                 # Conservative Neural-PLO integration (inspired by NDGPLO_BSA)
                 neural_blend_factor=0.5,  # Conservative blending factor (like NDGPLO_BSA)
                 adaptive_integration=True,  # Progressive integration over time
                 **kwargs):

        # Initialize parent OriginalPLO
        super().__init__(epoch, pop_size, **kwargs)

        # Neural network parameters
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Conservative Neural-PLO integration parameters (inspired by NDGPLO_BSA)
        self.neural_blend_factor = neural_blend_factor
        self.adaptive_integration = adaptive_integration

        # Neural network components
        self.dirnet = None
        self.optimizer = None
        self.criterion = CosineLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management for neural network training
        self.data = []

        # Training optimization parameters
        self.early_stopping_patience = 3
        self.min_loss_improvement = 1e-5
        self.global_step_counter = 0

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Collect direction data for neural network training from PLO operations.
        """
        # Determine which agent is better based on minmax
        if minmax == "min":
            if agent_x.target.fitness < agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x
        else:  # maxmax == "max"
            if agent_x.target.fitness > agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x

        # Collect direction data: from worse position toward better position
        direction = better_agent.solution - worse_agent.solution
        if np.linalg.norm(direction) > 1e-12:  # Only collect meaningful directions
            self.data.append({
                'start_pos': worse_agent.solution.copy(),
                'direction': direction.copy()
            })

        return better_agent.copy()

    def _predict_direction(self, current_position):
        """
        Predicts a unit vector direction from current position.
        Returns a direction vector that should be scaled as needed.
        """
        if not self.model_trained or self.dirnet is None:
            return None

        try:
            self.dirnet.eval()
            with torch.no_grad():
                # 1. Normalize input position
                pos_tensor = torch.tensor(current_position, dtype=torch.float32).to(self.device)
                pos_scale = torch.tensor((self.problem.ub - self.problem.lb) / 2.0, 
                                    dtype=torch.float32).to(self.device)
                pos_shift = torch.tensor((self.problem.ub + self.problem.lb) / 2.0, 
                                    dtype=torch.float32).to(self.device)
                norm_input = (pos_tensor - pos_shift) / pos_scale

                # 2. Get raw prediction (already near unit vector due to cosine loss training)
                predicted = self.dirnet(norm_input.unsqueeze(0)).squeeze(0)

                # 3. Convert to exact unit vector
                predicted_unit = predicted / torch.norm(predicted).clamp(min=1e-8)

                # 4. Return as numpy array (still a unit vector)
                return predicted_unit.cpu().numpy()

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Direction prediction error: {e}")
            return None

    def _train_neural_network(self, epoch):
        """
        Train the neural network with collected direction data from PLO operations.
        """
        if len(self.data) < self.min_data_for_training:
            return

        try:
            # Create train/validation split
            train_size = int(0.8 * len(self.data))
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

            train_data = [self.data[i] for i in indices[:train_size]]
            val_data = [self.data[i] for i in indices[train_size:]]

            # Create datasets and data loaders
            train_dataset = CustomDataset(train_data, self.problem.lb, self.problem.ub)
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)

            # Initialize network if needed
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims).to(self.device)

            # Setup optimizer
            self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Training loop with early stopping
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
                avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                # Early stopping check
                if avg_val_loss < best_val_loss - self.min_loss_improvement:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

                self.global_step_counter += 1

            # Mark model as trained
            self.model_trained = True

        except Exception as e:
            print(f"Error during neural network training: {e}")
            if self.logger:
                self.logger.error(f"Neural network training failed: {e}")

    def _conservative_neural_enhancement(self, current_solution, plo_movement, predicted_direction, progress_ratio):
        """
        Conservative Neural Enhancement: Simple blending inspired by NDGPLO_BSA success.

        Uses the same successful pattern as NDGPLO_BSA:
        - Progressive neural influence starting from 0%
        - Simple direction blending without complex scaling
        - Robust fallback to pure PLO
        """
        if predicted_direction is None:
            # Fallback to pure PLO movement
            return plo_movement

        # Conservative adaptive blending (like NDGPLO_BSA)
        if self.adaptive_integration:
            # Start with 0% neural influence, gradually increase
            blend_factor = self.neural_blend_factor * progress_ratio
        else:
            # Fixed conservative blending
            blend_factor = self.neural_blend_factor

        # Ensure conservative blending (max 80% neural influence like NDGPLO_BSA)
        blend_factor = np.clip(blend_factor, 0.0, 0.8)

        # Simple direction blending (NDGPLO_BSA pattern)
        # PLO direction: plo_movement - current_solution
        plo_direction = plo_movement - current_solution

        # Combine directions with conservative weighting
        combined_direction = ((1 - blend_factor) * plo_direction +
                            blend_factor * predicted_direction)

        # Generate enhanced movement
        enhanced_movement = current_solution + combined_direction

        return enhanced_movement

    def _get_adaptive_step(self, base_step=1):
        """Calculate adaptive step size with proper shape handling"""
        
        # Extract solutions and ensure numpy array
        solutions = np.array([agent.solution for agent in self.pop])
        
        # Ensure best_solution is numpy array and correct shape
        best_solution = np.array(self.g_best.solution)
        best_solution = best_solution.flatten()
        
        # Calculate statistics
        mu = np.mean(solutions, axis=0)
        sigma = np.std(solutions, axis=0)
        
        # Now calculate distance safely
        distance_to_best = np.abs(mu - best_solution)
        
        # Calculate adaptive step
        adaptive_step = base_step * (sigma / (distance_to_best + 1e-8))
        
        return adaptive_step


    def _apply_ndgplo_bsa_operator(self, current_agent, epoch, agent_idx):
        """
        Apply NDGPLO_BSA operator to a single agent.
        Uses BSA's proven mathematical formula with neural enhancement.
        """
        progress_ratio = epoch / self.epoch

        # Initialize historical population if not done
        if not hasattr(self, 'historical_population') or len(self.historical_population) == 0:
            self.historical_population = [agent.copy() for agent in self.pop]

        # Ensure historical population has enough agents
        if len(self.historical_population) <= agent_idx:
            self.historical_population.extend([agent.copy() for agent in self.pop])

        # BSA mutation components
        random_indices = self.generator.choice(len(self.pop), size=2, replace=False)
        P1 = self.pop[random_indices[0]].solution
        P2 = self.pop[random_indices[1]].solution
        oldP = self.historical_population[agent_idx].solution

        # Calculate F factor - BSA standard range
        # F = 3 * self.generator.random()  # BSA standard: F ∈ [0, 3]
        F = 3 * np.random.randn()  # BSA standard: F ∈ [0, 3]

        # BSA's PROVEN FORMULA: Mutant = oldP + F * (P1 - P2)
        # bsa_mutant = oldP + F * (P1 - P2)
        predicted_direction = self._predict_direction(current_agent.solution)
        adaptive_step = self._get_adaptive_step()
        bsa_mutant = current_agent.solution + predicted_direction * adaptive_step

        # Neural enhancement: Try to get predicted direction
        use_neural_guidance = (self.model_trained and
                             self.dirnet is not None and
                             self.generator.random() < self.prediction_usage_probability)

        if False and use_neural_guidance:
            predicted_direction = self._predict_direction(current_agent.solution)

            if False and predicted_direction is not None:
                # Apply conservative neural enhancement (NDGPLO_BSA pattern)
                enhanced_solution = self._conservative_neural_enhancement(
                    current_agent.solution, bsa_mutant, predicted_direction, progress_ratio)
                return self.correct_solution(enhanced_solution)
            else:
                # Fallback to standard BSA mutation
                return self.correct_solution(bsa_mutant)
        else:
            # Use standard BSA mutation
            return self.correct_solution(bsa_mutant)

    def _apply_original_plo_operator(self, current_agent, epoch, agent_idx):
        """
        Apply original PLO operators to a single agent.
        Uses PLO's Aurora Oval Walk, Levy Flight, and Particle Collision.
        """
        progress_ratio = epoch / self.epoch
        current_solution = current_agent.solution.copy()

        # Calculate mean position of the population
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)

        # Calculate adaptive weights for PLO components
        w1 = np.tanh((progress_ratio) ** 4)
        w2 = np.exp(-(2 * progress_ratio) ** 3)

        # E for particle collision
        E = np.sqrt(progress_ratio)

        # Generate random permutation for collision pairs
        A = self.generator.permutation(self.pop_size)

        # Aurora oval walk
        a = self.generator.uniform() / 2 + 1
        V = np.exp((1 - a) / 100 * epoch)
        LS = V

        # Levy flight movement component
        GS = self.levy(self.problem.n_dims) * (x_mean - current_solution +
                                               (self.problem.lb + self.generator.uniform(0, 1, self.problem.n_dims) *
                                                (self.problem.ub - self.problem.lb)) / 2)

        # Standard PLO position update
        plo_movement = current_solution + (w1 * LS + w2 * GS) * self.generator.uniform(0, 1, self.problem.n_dims)

        # Particle collision
        for j in range(self.problem.n_dims):
            if (self.generator.random() < 0.05) and (self.generator.random() < E):
                plo_movement[j] = current_solution[j] + np.sin(self.generator.random() * np.pi) * \
                                 (current_solution[j] - self.pop[A[agent_idx]].solution[j])

        return self.correct_solution(plo_movement)

    def _apply_hybrid_ndgplo_algorithm(self, epoch):
        """
        HYBRID NDGPLO: Probabilistic combination of NDGPLO_BSA and Original PLO

        Strategy:
        - When neural network is trained: 50% NDGPLO_BSA + 50% Original PLO
        - When neural network not trained: 100% Original PLO (fallback)

        This maintains NDGPLO_BSA's high performance while adding PLO's exploration diversity.
        """
        progress_ratio = epoch / self.epoch

        # Update BSA historical population periodically (for NDGPLO_BSA operators)
        if hasattr(self, 'historical_population') and len(self.historical_population) > 0:
            # BSA Selection-I: Decide whether to update historical population
            a = self.generator.random()
            b = self.generator.random()

            if a < b:
                # Update historical population with current population
                self.historical_population = [agent.copy() for agent in self.pop]

            # Permute (shuffle) the historical population - CRITICAL BSA STEP
            permuted_indices = self.generator.permutation(len(self.historical_population))
            self.historical_population = [self.historical_population[i] for i in permuted_indices]

        # === HYBRID POPULATION GENERATION ===
        new_population = []

        for idx in range(self.pop_size):
            current_agent = self.pop[idx]

            # Hybrid algorithm selection logic
            if self.model_trained:  # Only use hybrid approach when neural network is trained
                if self.generator.random() < 0.5:  # 50% probability
                    # Generate solution using NDGPLO_BSA approach (proven high performance)
                    new_solution = self._apply_ndgplo_bsa_operator(current_agent, epoch, idx)
                    operator_used = "NDGPLO_BSA"
                else:
                    # Generate solution using original PLO approach (exploration diversity)
                    new_solution = self._apply_original_plo_operator(current_agent, epoch, idx)
                    operator_used = "Original_PLO"
            else:
                # Fallback to pure original PLO when neural network not trained
                new_solution = self._apply_original_plo_operator(current_agent, epoch, idx)
                operator_used = "Original_PLO_Fallback"

            # Create new agent
            new_agent = self.generate_empty_agent(new_solution)
            new_agent.target = self.get_target(new_solution)

            # Collect direction data for neural network training
            self._collect_direction_data(current_agent, new_agent, self.problem.minmax)

            # Selection: Choose better agent (BSA-style selection)
            if new_agent.is_better_than(current_agent, self.problem.minmax):
                new_population.append(new_agent)
            else:
                new_population.append(current_agent)

        # === COMPLETE POPULATION REPLACEMENT (BSA-Style) ===
        # Replace ENTIRE population at once (maintains BSA's successful structure)
        self.pop = new_population

        # Update global best
        for agent in self.pop:
            if agent.is_better_than(self.g_best, self.problem.minmax):
                self.g_best = agent.copy()

        return True  # Hybrid NDGPLO algorithm completed successfully

    def evolve(self, epoch):
        """
        HYBRID NDGPLO: Probabilistic Combination of NDGPLO_BSA and Original PLO

        STRATEGIC APPROACH: Combines the proven high performance of NDGPLO_BSA
        with PLO's exploration characteristics through probabilistic selection.

        Algorithm Strategy:
        - When neural network is trained: 50% NDGPLO_BSA + 50% Original PLO
        - When neural network not trained: 100% Original PLO (fallback)

        Key Benefits:
        - Maintains NDGPLO_BSA's high performance (50% of solutions use proven BSA formula)
        - Adds PLO's exploration diversity (50% of solutions use PLO operators)
        - Uses same population-level processing that makes NDGPLO_BSA successful
        - Conservative neural enhancement strategy from NDGPLO_BSA

        Expected Performance: Equal to or better than pure NDGPLO_BSA
        """
        # Apply hybrid NDGPLO algorithm (NDGPLO_BSA + Original PLO)
        self._apply_hybrid_ndgplo_algorithm(epoch)

        # Periodic neural network training
        if (epoch % self.train_every == 0 and
            len(self.data) >= self.min_data_for_training):
            self._train_neural_network(epoch)

        # Limit data size to prevent memory issues
        if len(self.data) > 1024:
            self.data = self.data[-1024:]


# Using the mutant and complex directions to generate new solutions
class NDGPLO2(OriginalPLO):
    """
    IMPROVED Neural Direction Guided PLO - Conservative Neural Integration

    Based on successful patterns from NDGPLO_BSA:
    - Conservative neural integration with gradual progression
    - Simple, robust blending strategy inspired by NDGPLO_BSA success
    - Focus on enhancing PLO's core movement rather than all components
    - Reliable fallback to pure PLO when neural predictions unavailable

    Key Improvements:
    - Single conservative enhancement factor (like NDGPLO_BSA)
    - Progressive neural influence starting from 0%
    - Simplified integration without multi-component interference
    - Robust fallback mechanisms
    """

    def __init__(self, epoch, pop_size,
                 prediction_usage_probability=1.0,  # Probability of using neural predictions
                 min_data_for_training=256,
                 train_every=50,
                 n_grad_epochs=3,
                 batch_size=32,
                 hidden_nodes=8,
                 learning_rate=1e-3,
                 # Conservative Neural-PLO integration (inspired by NDGPLO_BSA)
                 neural_blend_factor=0.3,  # Conservative blending factor (like NDGPLO_BSA)
                 adaptive_integration=True,  # Progressive integration over time
                 **kwargs):

        # Initialize parent OriginalPLO
        super().__init__(epoch, pop_size, **kwargs)

        # Neural network parameters
        self.prediction_usage_probability = prediction_usage_probability
        self.min_data_for_training = min_data_for_training
        self.train_every = train_every
        self.n_grad_epochs = n_grad_epochs
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Conservative Neural-PLO integration parameters (inspired by NDGPLO_BSA)
        self.neural_blend_factor = neural_blend_factor
        self.adaptive_integration = adaptive_integration

        # Neural network components
        self.dirnet = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_trained = False

        # Data management for neural network training
        self.data = []

        # Training optimization parameters
        self.early_stopping_patience = 3
        self.min_loss_improvement = 1e-5
        self.global_step_counter = 0

    def _collect_direction_data(self, agent_x, agent_y, minmax: str = "min"):
        """
        Collect direction data for neural network training from PLO operations.
        """
        # Determine which agent is better based on minmax
        if minmax == "min":
            if agent_x.target.fitness < agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x
        else:  # maxmax == "max"
            if agent_x.target.fitness > agent_y.target.fitness:
                better_agent = agent_x
                worse_agent = agent_y
            else:
                better_agent = agent_y
                worse_agent = agent_x

        # Collect direction data: from worse position toward better position
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
        """
        if not self.model_trained or self.dirnet is None:
            return None

        try:
            self.dirnet.eval()
            with torch.no_grad():
                # Normalize input using position bounds
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

    def _train_neural_network(self, epoch):
        """
        Train the neural network with collected direction data from PLO operations.
        """
        if len(self.data) < self.min_data_for_training:
            return

        try:
            # Create train/validation split
            train_size = int(0.8 * len(self.data))
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)

            train_data = [self.data[i] for i in indices[:train_size]]
            val_data = [self.data[i] for i in indices[train_size:]]

            # Create datasets and data loaders
            train_dataset = CustomDataset(train_data, self.problem.lb, self.problem.ub)
            val_dataset = CustomDataset(val_data, self.problem.lb, self.problem.ub)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=False)

            # Initialize network if needed
            if self.dirnet is None:
                self.dirnet = DirNet(self.problem.n_dims, self.hidden_nodes,
                                   self.problem.n_dims).to(self.device)

            # Setup optimizer
            self.optimizer = optim.Adam(self.dirnet.parameters(), lr=self.learning_rate)

            # Training loop with early stopping
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
                avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                # Early stopping check
                if avg_val_loss < best_val_loss - self.min_loss_improvement:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

                self.global_step_counter += 1

            # Mark model as trained
            self.model_trained = True

        except Exception as e:
            print(f"Error during neural network training: {e}")
            if self.logger:
                self.logger.error(f"Neural network training failed: {e}")

    def _conservative_neural_enhancement(self, current_solution, plo_movement, predicted_direction, progress_ratio):
        """
        Conservative Neural Enhancement: Simple blending inspired by NDGPLO_BSA success.

        Uses the same successful pattern as NDGPLO_BSA:
        - Progressive neural influence starting from 0%
        - Simple direction blending without complex scaling
        - Robust fallback to pure PLO
        """
        if predicted_direction is None:
            # Fallback to pure PLO movement
            return plo_movement

        # Conservative adaptive blending (like NDGPLO_BSA)
        if self.adaptive_integration:
            # Start with 0% neural influence, gradually increase
            blend_factor = self.neural_blend_factor * progress_ratio
        else:
            # Fixed conservative blending
            blend_factor = self.neural_blend_factor

        # Ensure conservative blending (max 80% neural influence like NDGPLO_BSA)
        blend_factor = np.clip(blend_factor, 0.0, 0.8)

        # Simple direction blending (NDGPLO_BSA pattern)
        # PLO direction: plo_movement - current_solution
        plo_direction = plo_movement - current_solution

        # Combine directions with conservative weighting
        combined_direction = ((1 - blend_factor) * plo_direction +
                            blend_factor * predicted_direction)

        # Generate enhanced movement
        enhanced_movement = current_solution + combined_direction

        return enhanced_movement

    def _apply_ndgplo_bsa_operator(self, current_agent, epoch, agent_idx):
        """
        Apply NDGPLO_BSA operator to a single agent.
        Uses BSA's proven mathematical formula with neural enhancement.
        """
        progress_ratio = epoch / self.epoch

        # Initialize historical population if not done
        if not hasattr(self, 'historical_population') or len(self.historical_population) == 0:
            self.historical_population = [agent.copy() for agent in self.pop]

        # Ensure historical population has enough agents
        if len(self.historical_population) <= agent_idx:
            self.historical_population.extend([agent.copy() for agent in self.pop])

        # BSA mutation components
        random_indices = self.generator.choice(len(self.pop), size=2, replace=False)
        P1 = self.pop[random_indices[0]].solution
        P2 = self.pop[random_indices[1]].solution
        oldP = self.historical_population[agent_idx].solution

        # Calculate F factor - BSA standard range
        F = 3 * self.generator.random()  # BSA standard: F ∈ [0, 3]

        # BSA's PROVEN FORMULA: Mutant = oldP + F * (P1 - P2)
        # bsa_mutant = oldP + F * (P1 - P2)

        # Neural enhancement: Try to get predicted direction
        use_neural_guidance = (self.model_trained and
                             self.dirnet is not None and
                             self.generator.random() < self.prediction_usage_probability)

        if True or use_neural_guidance:
            predicted_direction = self._predict_direction(current_agent.solution)

            # bsa_mutant = oldP + F * (P1 - P2)
            bsa_mutant = oldP + F * predicted_direction

            if True or predicted_direction is not None:
                # Apply conservative neural enhancement (NDGPLO_BSA pattern)
                enhanced_solution = self._conservative_neural_enhancement(
                    current_agent.solution, bsa_mutant, predicted_direction, progress_ratio)
                return self.correct_solution(enhanced_solution)
            else:
                # Fallback to standard BSA mutation
                return self.correct_solution(bsa_mutant)
        else:
            # Use standard BSA mutation
            return self.correct_solution(bsa_mutant)

    def _apply_original_plo_operator(self, current_agent, epoch, agent_idx):
        """
        Apply original PLO operators to a single agent.
        Uses PLO's Aurora Oval Walk, Levy Flight, and Particle Collision.
        """
        progress_ratio = epoch / self.epoch
        current_solution = current_agent.solution.copy()

        # Calculate mean position of the population
        x_mean = np.mean([agent.solution for agent in self.pop], axis=0)

        # Calculate adaptive weights for PLO components
        w1 = np.tanh((progress_ratio) ** 4)
        w2 = np.exp(-(2 * progress_ratio) ** 3)

        # E for particle collision
        E = np.sqrt(progress_ratio)

        # Generate random permutation for collision pairs
        A = self.generator.permutation(self.pop_size)

        # Aurora oval walk
        a = self.generator.uniform() / 2 + 1
        V = np.exp((1 - a) / 100 * epoch)
        LS = V

        # Levy flight movement component
        GS = self.levy(self.problem.n_dims) * (x_mean - current_solution +
                                               (self.problem.lb + self.generator.uniform(0, 1, self.problem.n_dims) *
                                                (self.problem.ub - self.problem.lb)) / 2)

        # Standard PLO position update
        plo_movement = current_solution + (w1 * LS + w2 * GS) * self.generator.uniform(0, 1, self.problem.n_dims)

        # Particle collision
        for j in range(self.problem.n_dims):
            if (self.generator.random() < 0.05) and (self.generator.random() < E):
                plo_movement[j] = current_solution[j] + np.sin(self.generator.random() * np.pi) * \
                                 (current_solution[j] - self.pop[A[agent_idx]].solution[j])

        return self.correct_solution(plo_movement)

    def _apply_hybrid_ndgplo_algorithm(self, epoch):
        """
        HYBRID NDGPLO: Probabilistic combination of NDGPLO_BSA and Original PLO

        Strategy:
        - When neural network is trained: 50% NDGPLO_BSA + 50% Original PLO
        - When neural network not trained: 100% Original PLO (fallback)

        This maintains NDGPLO_BSA's high performance while adding PLO's exploration diversity.
        """
        progress_ratio = epoch / self.epoch

        # Update BSA historical population periodically (for NDGPLO_BSA operators)
        if hasattr(self, 'historical_population') and len(self.historical_population) > 0:
            # BSA Selection-I: Decide whether to update historical population
            a = self.generator.random()
            b = self.generator.random()

            if a < b:
                # Update historical population with current population
                self.historical_population = [agent.copy() for agent in self.pop]

            # Permute (shuffle) the historical population - CRITICAL BSA STEP
            permuted_indices = self.generator.permutation(len(self.historical_population))
            self.historical_population = [self.historical_population[i] for i in permuted_indices]

        # === HYBRID POPULATION GENERATION ===
        new_population = []

        for idx in range(self.pop_size):
            current_agent = self.pop[idx]

            # Hybrid algorithm selection logic
            if self.model_trained:  # Only use hybrid approach when neural network is trained
                if self.generator.random() < 0.5:  # 50% probability
                    # Generate solution using NDGPLO_BSA approach (proven high performance)
                    new_solution = self._apply_ndgplo_bsa_operator(current_agent, epoch, idx)
                    operator_used = "NDGPLO_BSA"
                else:
                    # Generate solution using original PLO approach (exploration diversity)
                    new_solution = self._apply_original_plo_operator(current_agent, epoch, idx)
                    operator_used = "Original_PLO"
            else:
                # Fallback to pure original PLO when neural network not trained
                new_solution = self._apply_original_plo_operator(current_agent, epoch, idx)
                operator_used = "Original_PLO_Fallback"

            # Create new agent
            new_agent = self.generate_empty_agent(new_solution)
            new_agent.target = self.get_target(new_solution)

            # Collect direction data for neural network training
            self._collect_direction_data(current_agent, new_agent, self.problem.minmax)

            # Selection: Choose better agent (BSA-style selection)
            if new_agent.is_better_than(current_agent, self.problem.minmax):
                new_population.append(new_agent)
            else:
                new_population.append(current_agent)

        # === COMPLETE POPULATION REPLACEMENT (BSA-Style) ===
        # Replace ENTIRE population at once (maintains BSA's successful structure)
        self.pop = new_population

        # Update global best
        for agent in self.pop:
            if agent.is_better_than(self.g_best, self.problem.minmax):
                self.g_best = agent.copy()

        return True  # Hybrid NDGPLO algorithm completed successfully

    def evolve(self, epoch):
        """
        HYBRID NDGPLO: Probabilistic Combination of NDGPLO_BSA and Original PLO

        STRATEGIC APPROACH: Combines the proven high performance of NDGPLO_BSA
        with PLO's exploration characteristics through probabilistic selection.

        Algorithm Strategy:
        - When neural network is trained: 50% NDGPLO_BSA + 50% Original PLO
        - When neural network not trained: 100% Original PLO (fallback)

        Key Benefits:
        - Maintains NDGPLO_BSA's high performance (50% of solutions use proven BSA formula)
        - Adds PLO's exploration diversity (50% of solutions use PLO operators)
        - Uses same population-level processing that makes NDGPLO_BSA successful
        - Conservative neural enhancement strategy from NDGPLO_BSA

        Expected Performance: Equal to or better than pure NDGPLO_BSA
        """
        # Apply hybrid NDGPLO algorithm (NDGPLO_BSA + Original PLO)
        self._apply_hybrid_ndgplo_algorithm(epoch)

        # Periodic neural network training
        if (epoch % self.train_every == 0 and
            len(self.data) >= self.min_data_for_training):
            self._train_neural_network(epoch)

        # Limit data size to prevent memory issues
        if len(self.data) > 1024:
            self.data = self.data[-1024:]