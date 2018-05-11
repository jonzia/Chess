# ----------------------------------------------------
# Test Bench for Chess AI v1.0.2
# Created By: Jonathan Zia
# Last Edited: Thursday, May 10 2018
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

# This program compares the performance of the specified trained model versus
# a second model for validation purposes. The program calculates wins/losses
# of the trained model versus a model that follows a random policy.

# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
# Value Function Approximator Training
NUM_TESTING	= 100		# Number of testing steps
HIDDEN_UNITS = 100		# Number of hidden units
BATCH_SIZE = 5			# Batch size

# Simulation Parameters
MAX_MOVES = 100			# Maximum number of moves
EPSILON = 0.0			# Defining epsilon for e-greedy policy (0 for testing -> greedy policy)

# Load File
LOAD_FILE = True 		# Load trained model from saved checkpoint (True for testing)
VISUALIZE = True		# Select True to visualize games and False to suppress game output
PRINT = True			# Select True to print moves as text and False to suppress printing
ALGEBRAIC = True		# Specify long algebraic notation (True) or descriptive text (False)


# ----------------------------------------------------
# Data Paths
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "D:\\"
with tf.name_scope("Model_Data"):		# Model save/load paths
	load_path = os.path.join(dir_name, "checkpoints/model")			# Model load path
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "output")
with tf.name_scope("Output_Data"):		# Output data filenames (.txt)
	# These .txt files will contain loss data for Matlab analysis
	outcome_file = os.path.join(dir_name, "outcomes.txt")


# ----------------------------------------------------
# User-Defined Methods
# ----------------------------------------------------
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

def move_piece(piece,move_index,player,pieces,switch_player=False,print_move=False,algebraic=True):
	"""
	Perform specified move
	Returns: Void
	"""
	if player == 'white':
		pieces[piece].move(move_index,pieces,print_move=print_move,algebraic=algebraic)
	else:
		pieces[piece+16].move(move_index,pieces,print_move=print_move,algebraic=algebraic)

	if switch_player:
		if player == 'white':
			player = 'black'
		else:
			player = 'white'
		return player

def generate_outcome(batch_size,max_moves,epsilon,visualize,print_move,algebraic):
	"""
	Generating feature and target batches
	Returns: (1) feature batch, (2) label batch, (3) visualize board?, (4) print move?, (5) print algebraic notation?
	"""

	# Generates training data based on batches of full-depth Monte-Carlo simulations
	# performing epsilon-greedy policy evalutaion.

	# Initialize placeholder for outcome batch
	outcome_batch = []

	# Loop through batch steps
	for batch_step in range(0,batch_size):

		# Print beginning of game notification for visualization
		if visualize or print_move:
			print("\n----------BEGIN GAME----------")

		# ----------------------------------------------------
		# Initialize Board State
		# ----------------------------------------------------
		# Create placeholders for board states and return for each state
		all_states = []
		all_returns = []

		# Generating board parameters
		pieces, initial_state, player, move = initialize_board(random=False, keep_prob=1.0)
		point_diff_0 = s.points(pieces)


		# ----------------------------------------------------
		# Monte Carlo Simulations
		# ----------------------------------------------------
		# Run Monte Carlo Simulation until terminal event(s):
		# Terminal events: Kings.is_active == False or move_counter > MAX_MOVES
		while pieces[4].is_active and pieces[28].is_active and move < max_moves:

			# Obtain board state
			if move == 0:
				board_state = initial_state
			else:
				board_state = s.board_state(pieces)

			# Visualize board state
			if visualize:
				visualize_board(pieces,player,move)

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
						expected_return = sess.run(predictions, feed_dict={inputs: np.reshape(temp_board_state,(1,768))})
						# Write estimated return to return_array
						return_array[i,j] = expected_return


			# ----------------------------------------------------
			# Policy
			# ----------------------------------------------------
			# Player white chooses greedy policy, player black chooses random policy
			# For player black, choose a random action
			if player == 'black':
				while True:
					# If the action is valid...
					piece_index = r.randint(0,15)
					move_index = r.randint(0,55)
					if return_array[piece_index,move_index] != 0:
						# Perform move and update player
						player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=print_move,algebraic=algebraic)
						break
			# Else, act greedy w.r.t. expected return
			else:
				# Identify indices of maximum return (white) or minimum return (black)
				move_choice = np.nonzero(return_array.max() == return_array)
				piece_index = move_choice[0][0]
				move_index = move_choice[1][0]
				# Perform move and update player
				player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=print_move,algebraic=algebraic)

			# Increment move counter
			move += 1

		# Print end of game notification for visualization
		if visualize or print_move:
			print("----------END OF GAME----------")

		# Append outcome
		# If player white won the game...
		if all_returns[0] > 0:
			outcome_batch.append(1)		# Return 1
		# Else, for a draw...
		elif all_returns[0] == 0:
			outcome_batch.append(0)		# Return 0 
		# Else, if player black won the game...
		else:
			outcome_batch.append(-1)	# Return -1

	# Return outcome batch
	outcome_batch = np.array(outcome_batch)
	return outcome_batch


# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------
# Create placeholders for inputs and target values
# Input dimensions: 8 x 8 x 12
# Target dimensions: 1 x 1
inputs = tf.placeholder(tf.float32,[None,768],name='Inputs')
targets = tf.placeholder(tf.float32,shape=(None,1),name='Targets')


# ----------------------------------------------------
# Implementing Feedforward NN
# ----------------------------------------------------
# First fully-connected layer
hidden1 = tf.contrib.layers.fully_connected(inputs,num_outputs=HIDDEN_UNITS)

# Second fully-connected layer
hidden2 = tf.contrib.layers.fully_connected(hidden1,num_outputs=HIDDEN_UNITS)

# Output layer
predictions = tf.contrib.layers.fully_connected(hidden2,num_outputs=1,activation_fn=None)


# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
saver = tf.train.Saver()	# Instantiate Saver class
outcomes = []				# Initialize placeholder for outcomes
with tf.Session() as sess:

	# Create Tensorboard graph
	#writer = tf.summary.FileWriter(filewriter_path, sess.graph)
	#merged = tf.summary.merge_all()

	# Restore saved session
	saver.restore(sess, load_path)

	# Obtain start time
	start_time = t.time()

	# For each training step, generate a random board
	for step in range(0,NUM_TESTING):

		# Run game and determine outcome
		outcome = generate_outcome(batch_size=BATCH_SIZE,max_moves=MAX_MOVES,epsilon=EPSILON,visualize=VISUALIZE,print_move=PRINT,algebraic=ALGEBRAIC)
		outcomes.append(outcome)


		# Conditional statement for calculating time remaining and percent completion
		if step % 1 == 0:

			# Report percent completion
			p_completion = 100*step/NUM_TESTING
			print("\nPercent Completion: %.3f%%" % p_completion)

			# Print time remaining
			avg_elapsed_time = (t.time() - start_time)/(step+1)
			sec_remaining = avg_elapsed_time*(NUM_TESTING-step)
			min_remaining = round(sec_remaining/60)
			print("Time Remaining: %d minutes" % min_remaining)

			# Print mean outcome
			print(outcome)
			print("Mean Outcome: %.3f" % np.mean(outcomes))

	# Write outcomes to file
	outcomes = np.array(outcomes)
	with open(outcome_file, 'a') as file_object:
		np.savetxt(file_object, outcomes)

	# Close the writer
	# writer.close()