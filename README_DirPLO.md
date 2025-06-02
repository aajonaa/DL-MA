# DirPLO on CEC2017 Benchmark Functions

This repository contains the implementation of DirPLO (Direction-guided Polar Lights Optimizer) and tools to run it on CEC2017 benchmark functions.

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install numpy scipy pandas matplotlib torch tensorboard scikit-learn
```

### 2. Run the Demo

For a quick test, run the demo script:

```bash
python demo_dirplo.py
```

This will run a comparison between DirPLO and OriginalPLO on a few CEC2017 functions.

### 3. Use the Jupyter Notebook

For comprehensive experiments, open the Jupyter notebook:

```bash
jupyter notebook run_dirplo.ipynb
```

## 📁 File Structure

- `PLO.py` - Contains the DirPLO, OriginalPLO, and KLEPLO implementations
- `optimizer.py` - Base optimizer class
- `utils/` - Utility modules (Problem, Agent, etc.)
- `cec2017/` - CEC2017 benchmark functions
- `run_dirplo.ipynb` - Comprehensive Jupyter notebook for experiments
- `demo_dirplo.py` - Quick demo script
- `README_DirPLO.md` - This file

## 🔧 How to Use

### Basic Usage

```python
from PLO import DirPLO
from utils import Problem, FloatVar
import cec2017.functions as cec2017

# Create a CEC2017 problem wrapper
class CEC2017Problem(Problem):
    def __init__(self, func_num, n_dims=30):
        self.cec_func = cec2017.all_functions[func_num - 1]
        bounds = FloatVar(lb=[-100.0] * n_dims, ub=[100.0] * n_dims)
        super().__init__(bounds, minmax='min', name=f'CEC2017_F{func_num}')
    
    def obj_func(self, x):
        x_2d = x.reshape(1, -1)
        return float(self.cec_func(x_2d)[0])

# Create problem and optimizer
problem = CEC2017Problem(func_num=1, n_dims=30)
optimizer = DirPLO(epoch=1000, pop_size=50)

# Solve
best_agent = optimizer.solve(problem, seed=42)
print(f"Best fitness: {best_agent.target.fitness}")
```

### Available Algorithms

1. **OriginalPLO** - The original Polar Lights Optimizer
2. **DirPLO** - Direction-guided PLO with neural network guidance
3. **KLEPLO** - Knowledge Learning Enhanced PLO

### CEC2017 Functions

The code supports all 30 CEC2017 benchmark functions (F1-F30):

- F1-F10: Simple functions
- F11-F20: Hybrid functions  
- F21-F30: Composition functions

## 📊 Experiment Configuration

The notebook allows you to configure:

- **Dimensions**: Problem dimensions to test (e.g., [10, 30, 50])
- **Functions**: Which CEC2017 functions to test (e.g., [1, 3, 4, 5])
- **Runs**: Number of independent runs for statistical analysis
- **Epochs**: Maximum number of iterations
- **Population Size**: Number of solutions in the population

## 📈 Results Analysis

The notebook provides:

1. **Statistical Summary**: Mean, std, min fitness values
2. **Convergence Plots**: Fitness vs iteration curves
3. **Performance Comparison**: DirPLO vs OriginalPLO
4. **Statistical Tests**: Wilcoxon signed-rank test for significance

## 🎯 Key Features of DirPLO

1. **Neural Network Guidance**: Uses a direction network to learn successful search directions
2. **Adaptive Training**: Trains the network during optimization when sufficient data is available
3. **Hybrid Approach**: Combines traditional PLO with learned guidance
4. **Data Collection**: Automatically collects successful moves for training

## 🔍 Understanding the Results

- **Positive improvement %**: DirPLO performed better
- **Negative improvement %**: OriginalPLO performed better
- **Statistical significance**: 
  - `***` p < 0.001 (highly significant)
  - `**` p < 0.01 (significant)
  - `*` p < 0.05 (marginally significant)
  - `ns` not significant

## ⚠️ Important Notes

1. **Training Phase**: DirPLO needs time to collect data and train the network
2. **Problem Dependency**: Performance may vary across different problem types
3. **Computational Cost**: DirPLO has additional overhead for neural network training
4. **Hyperparameters**: May need tuning for optimal performance on specific problems

## 🐛 Troubleshooting

### Import Errors
If you get import errors, restart your Jupyter kernel or run:

```python
import sys
import importlib
importlib.invalidate_caches()

# Remove cached modules
modules_to_remove = [name for name in sys.modules.keys() 
                    if name.startswith('utils') or name == 'PLO' or name == 'optimizer']
for module in modules_to_remove:
    if module in sys.modules:
        del sys.modules[module]
```

### Missing Dependencies
Install missing packages:

```bash
pip install tensorboard  # For DirPLO logging
pip install scipy       # For statistical analysis
pip install pandas      # For data analysis
```

## 📚 References

- CEC2017 Benchmark Suite
- Polar Lights Optimizer (PLO)
- Direction-guided optimization techniques

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or additional features!
