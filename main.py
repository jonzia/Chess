# ----------------------------------------------------
# Chess AI
# Created By: Jonathan Zia
# Last Edited: Saturday, March 17 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import numpy as np
import pieces as p
import random as r
import state as s
import time as t
import copy as c
import math
import os


# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
# Value Function Approximator Training
NUM_TRAINING = 1000		# Number of training steps
HIDDEN_UNITS = 100		# Number of hidden units
LEARNING_RATE = 0.001	# Learning rate
BATCH_SIZE = 5			# Batch size (pending)

# Simulation Parameters
MAX_MOVES = 100			# Maximum number of moves for Monte Carlo
EPSILON = 0.2			# Defining epsilon for e-greedy policy

# Load File
LOAD_FILE = False 		# Load initial value model from saved checkpoint?


# ----------------------------------------------------
# Data Paths
# ----------------------------------------------------
# Specify filenames
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


# ----------------------------------------------------
# User-Defined Methods
# ----------------------------------------------------
def init_values(shape):
	"""
	Initialize Weight and Bias Matrices
	Returns: Tensor of shape "shape" w/ normally-distributed values
	"""
	temp = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(temp)

def initialize_board(random=False, keep_prob=1.0):
	"""
	Initialize Game Board
	Returns: Game board state parameters
	"""
	# Initialize board pieces
	pieces = s.initialize_pieces(random=random,keep_prob=keep_prob)
	# Initialize state space 
	board_state = s.board_state(pieces)
	# Initialize current player:
	if random:
		if r.randint(0,1) == 1:
			player = 'white'
		else:
			player = 'black'
	else:
		player = 'white'
	# Initialize move counter:
	move = 0

	# Return values
	return pieces, board_state, player, move

def visualize_board(pieces, player, move):
	"""
	Visualize Game Board
	Returns: Void
	"""
	print("\nCurrent Board at Move " + str(move) + " for Player " + player)
	print(s.visualize_state(pieces))

def move_piece(piece,move_index,player,pieces,switch_player=False,print_move=False):
	"""
	Perform specified move
	Returns: Void
	"""
	if player == 'white':
		pieces[piece].move(move_index,pieces,print_move=print_move)
	else:
		pieces[piece+16].move(move_index,pieces,print_move=print_move)

	if switch_player:
		if player == 'white':
			player = 'black'
		else:
			player = 'white'
		return player


# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------
# Create placeholders for inputs and target values
# Input dimensions: 8 x 8 x 12
# Target dimensions: 1 x 1
inputs = tf.placeholder(tf.float32,[8,8,12],name='Inputs')
targets = tf.placeholder(tf.float32,shape=(),name='Targets')


# ----------------------------------------------------
# Initializing Feedforward NN
# ----------------------------------------------------
# First fully-connected layer
with tf.name_scope("FCN_1"):
	W_1 = init_values([768, HIDDEN_UNITS])
	tf.summary.histogram('Weights_1',W_1)
	b_1 = init_values([HIDDEN_UNITS])
	tf.summary.histogram('Biases_1',b_1)

# Second fully-connected layer
with tf.name_scope("FCN_2"):
	W_2 = init_values([HIDDEN_UNITS,1])
	tf.summary.histogram('Weights_2',W_2)
	b_2 = init_values([1,1])
	tf.summary.histogram('Biases_2',b_2)

# ----------------------------------------------------
# Evaluating Feedforward NN
# ----------------------------------------------------
# Obtain hidden values by performing sigmoid((weights)*(output)+(biases))
with tf.name_scope("Hidden_Layer"):
	hidden = tf.sigmoid(tf.matmul(tf.transpose(tf.reshape(inputs,[-1,1])), W_1) + b_1)
# Obtain predictions by performing (weights)*(output)+(biases)
with tf.name_scope("Output_Layer"):
	predictions = tf.matmul(hidden, W_2) + b_2

# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating mean squared error of predictions and targets
loss = tf.losses.mean_squared_error(labels=tf.reshape(targets,[1,1]), predictions=predictions)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()	# Instantiate Saver class
t_loss = []	# Placeholder for training loss values
with tf.Session() as sess:

	# Create Tensorboard graph
	writer = tf.summary.FileWriter(filewriter_path, sess.graph)
	merged = tf.summary.merge_all()

	# If there is a model checkpoint saved, load the checkpoint. Else, initialize variables.
	if LOAD_FILE:
		# Restore saved session
		saver.restore(sess, load_path)
	else:
		# Initialize the variables
		sess.run(init)

	# Obtain start time
	start_time = t.time()

	# For each training step, generate a random board
	for step in range(0,NUM_TRAINING):


		# ----------------------------------------------------
		# Initialize Board State
		# ----------------------------------------------------
		# Create placeholders for board states and return for each state
		all_states = []
		all_returns = []

		# Generating board parameters
		pieces, initial_state, player, move = initialize_board(random=True, keep_prob=0.8)
		point_diff_0 = s.points(pieces)


		# ----------------------------------------------------
		# Monte Carlo Simulations
		# ----------------------------------------------------
		# Run Monte Carlo Simulation until terminal event(s):
		# Terminal events: Kings.is_active == False or move_counter > MAX_MOVES
		while pieces[4].is_active and pieces[28].is_active and move < MAX_MOVES:

			# Obtain board state
			if move == 0:
				board_state = initial_state
			else:
				board_state = s.board_state(pieces)

			# Visualize board state
			# visualize_board(pieces,player,move)

			# Obtain current point differential
			net_diff = s.points(pieces) - point_diff_0
			point_diff_0 = s.points(pieces)
			
			# Append initial board state to all_states
			all_states.append(board_state)
			# Add net_diff to all existing returns
			for i in range(0,len(all_returns)):
				all_returns[i] += net_diff
			# Append 0 to end of all_returns representing return for current state
			all_returns.append(0)

			# Obtain action space
			action_space = s.action_space(pieces,player)


			# ----------------------------------------------------
			# Value Function Approximation
			# ----------------------------------------------------
			# For each action in the action space, obtain subsequent board space
			# and calculate estimated return with the partially-trained approximator

			# Create placeholder for expected return values
			return_array = np.zeros((16,56))

			# For each possible move...
			for i in range(0,16):
				for j in range(0,56):
					# If the move is legal...
					if action_space[i,j] == 1:

						# Perform move and obtain temporary board state
						temp_pieces = c.deepcopy(pieces)				# Reset temporary pieces variable
						move_piece(i,j,player,temp_pieces)				# Perform temporary move
						temp_board_state = s.board_state(temp_pieces)	# Obtain temporary state

						# With temporary state, calculate expected return
						expected_return = sess.run(predictions, feed_dict={inputs: temp_board_state})

						# Write estimated return to return_array
						return_array[i,j] = expected_return


			# ----------------------------------------------------
			# Epsilon-Greedy Policy
			# ----------------------------------------------------
			# With probability epsilon, choose a random action
			if r.random() > EPSILON:
				while True:
					# If the action is valid...
					piece_index = r.randint(0,15)
					move_index = r.randint(0,55)
					if return_array[piece_index,move_index] != 0:
						# Perform move and update player
						player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=False)
						break
			# Else, act greedy w.r.t. expected return
			else:
				# Identify indices of maximum return
				move_choice = np.nonzero(return_array.max() == return_array)
				piece_index = move_choice[0][0]
				move_index = move_choice[1][0]
				# Perform move and update player
				player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=False)

			# Increment move counter
			move += 1

		# ----------------------------------------------------
		# Optimize Model for Current Simulation
		# ----------------------------------------------------			
		# Print step
		print("\nOptimizing at step", step)
		# Run optimizer, loss, and predicted error ops in graph
		predictions_, targets_, _, loss_ = sess.run([predictions, targets, optimizer, loss], feed_dict={inputs: np.array(all_states[0]), targets: np.array(all_returns[0])})

		# Record loss
		t_loss.append(loss_)

		# Save and overwrite the session at each training step
		saver.save(sess, save_path)
		# Writing summaries to Tensorboard at each training step
		summ = sess.run(merged)
		writer.add_summary(summ,step)

		# Conditional statement for calculating time remaining, percent completion, and loss
		if step % 10 == 0:

			# Report loss
			print("\nTrain loss at step", step, ":", loss_)

			# Report percent completion
			p_completion = 100*step/NUM_TRAINING
			print("Percent completion: %.3f%%" % p_completion)

			# Print time remaining
			avg_elapsed_time = (t.time() - start_time)/(step+1)
			sec_remaining = avg_elapsed_time*(NUM_TRAINING-step)
			min_remaining = round(sec_remaining/60)
			print("Time Remaining: %d minutes" % min_remaining)

	# Write training loss to file
	t_loss = np.array(t_loss)
	with open(training_loss, 'a') as file_object:
		np.savetxt(file_object, t_loss)

	# Close the writer
	writer.close()