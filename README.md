# Chess RL v1.0.1

## Overview
Tensorflow program that learns to play chess via Reinforcement Learning. Check out the [design blog](https://www.jonzia.me/projects/chess-reinforcement-learning)!

## Description
This program learns to play chess via reinforcement learning. The action-value functions are learned by training a neural network on the total return of randomly-initialized board states, determined by Monte Carlo simulations. The program follows an epsilon-greedy policy based on the most current action-value function approximations. As of v1.0.1, each training step is trained on batches of full-depth Monte Carlo simulations. The model architecture has one hidden layer, though this can be easily expanded or even updated to a convolutional architecture (to be included in a future release).

The game's basic rules are encoded in *pieces.py* and the board state parameters are defined in *state.py*. Once a proper action-value function is converged upon, it can be implemented with a greedy policy for purposes of gameplay. The program *test_bench.py* is included for validating trained model performance against a benchmark policy.

![Tensorboard Graph v1.0.0](https://raw.githubusercontent.com/jonzia/Chess_RL/master/Media/Graph_100.PNG)

## To Run
1. Install [Tensorflow](https://www.tensorflow.org/) **(1)**
2. Set user-defined parameters in *main.py*.
```python
# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
# Value Function Approximator Training
NUM_TRAINING = 100	# Number of training steps
HIDDEN_UNITS = 100	# Number of hidden units
LEARNING_RATE = 0.001	# Learning rate
BATCH_SIZE = 5		# Batch size
# Simulation Parameters
MAX_MOVES = 100		# Maximum number of moves for Monte Carlo
EPSILON = 0.2		# Defining epsilon for e-greedy policy
VISUALIZE = True	# Select True to visualize games and False to suppress game output
PRINT = True		# Select True to print moves as text and False to suppress printing
ALGEBRAIC = True	# Specify long algebraic notation (True) or descriptive text (False)
# Load File
LOAD_FILE = False 	# Load initial value model from saved checkpoint?
```
3. Set file paths in *main.py*.
```python
# Root directory:
dir_name = "D:\\Documents"
with tf.name_scope("Model_Data"):		# Model save/load paths
	load_path = os.path.join(dir_name, "checkpoints\\model")			# Load previous model
	save_path = os.path.join(dir_name, "checkpoints\\model")			# Save model at each step
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "output")
with tf.name_scope("Output_Data"):		# Output data filenames (.txt)
	# These .txt files will contain loss data for Matlab analysis
	training_loss = os.path.join(dir_name, "training_loss.txt")
```
4. Run *main.py*. **(2)**
5. (Optional) Run [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to visualize learning. **(3)**
6. (Optional) Run *test_bench.py* to compare model performance against a benchmark. At the current time, the benchmark is a random policy, however future releases will allow the user to load other benchmark models. **(4)**

## Update Log
_v1.0.2_: Included support for game visualization and move printing in chess or longhand notation.

_v1.0.1_: Bug fixes and support for large training batches. Added test bench program for analysis.

_v1.0.0_: Beta version.

### Notes
**(1)** This program was built on Python 3.6 and Tensorflow 1.5.

**(2)** The terminal display includes the current step, training loss, percent completion, and time remaining. Training games may be visualized based on user-defined settings above. The current model is saved at each time step.

**(3)** Upon completion of training, training loss at each step is written to an output .txt file for analysis.

**(4)** This program outputs training progress and mean outcome in the terminal (where outcomes are -1 for loss, 0 for draw, 1 for win). This information is saved to an output .txt file for subsequent statistical analysis. Testing games may be visualized based on user-defined settings above.
