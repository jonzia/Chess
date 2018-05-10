# ----------------------------------------------------
# Chess AI v1.0.3
# Created By: Jonathan Zia
# Last Edited: Tuesday, March 21 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import numpy as np
import pieces as p
import random as r
import state as s
import time as t
import copy as c
import argparse
import math
import os

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

def generate_game(batch_size,max_moves,epsilon,visualize,print_move,algebraic):
	"""
	Generating feature and target batches
	Returns: (1) feature batch, (2) label batch
	"""

	# Generates training data based on batches of full-depth Monte-Carlo simulations
	# performing epsilon-greedy policy evalutaion.

	# Initialize placeholders for batches
	feature_batches = []
	label_batches = []

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
		pieces, initial_state, player, move = initialize_board(random=True, keep_prob=0.8)
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
			# Epsilon-Greedy Policy
			# ----------------------------------------------------
			# With probability epsilon, choose a random action
			if r.random() < epsilon:
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
				if player == 'white':
					# Find the indices of the maximum nonzero value
					maxval = np.max(return_array[np.nonzero(return_array)])
					maxdim = np.argwhere(return_array==maxval)
					piece_index = maxdim[0][0]	# Maximum (row)
					move_index = maxdim[0][1]	# Maximum (column)
				else:
					# Find the indices of the minimum nonzero value
					minval = np.min(return_array[np.nonzero(return_array)])
					mindim = np.argwhere(return_array==minval)
					piece_index = mindim[0][0]	# Maximum (row)
					move_index = mindim[0][1]	# Maximum (column)
				# Perform move and update player
				player = move_piece(piece_index,move_index,player,pieces,switch_player=True,print_move=print_move,algebraic=algebraic)
			# Increment move counter
			move += 1

		# Print end of game notification for visualization
		if visualize or print_move:
			print("----------END OF GAME----------")

		feature_batches.append(initial_state)
		label_batches.append(all_returns[0])

	# Return features and labels
	feature_batches = np.array(feature_batches)
	label_batches = np.array(label_batches)
	return feature_batches, label_batches


if __name__ == "__main__":

	# ----------------------------------------------------
	# Parsing Console Arguments
	# ----------------------------------------------------
	
	# Create parser object
	parser = argparse.ArgumentParser()

	parser.add_argument("-t","--trainsteps", help="Number of training steps (Default 1000)", type=int)
	parser.add_argument("-u","--hidunits", help="Number of hidden units (Default 100)", type=int)
	parser.add_argument("-r","--learnrate", help="Learning rate (Default 0.001)", type=float)
	parser.add_argument("-b","--batchsize", help="Batch size (Default 32)", type=int)
	parser.add_argument("-m","--maxmoves", help="Maximum moves for MC simulations (Default 100)", type=int)
	parser.add_argument("-e","--epsilon", help="Epsilon-greedy policy evaluation (Default 0.2)", type=float)
	parser.add_argument("-v","--visualize", help="Visualize game board? (Default False)", type=bool)
	parser.add_argument("-p","--print", help="Print moves? (Default False)", type=bool)
	parser.add_argument("-a","--algebraic", help="Print moves in algebraic notation? (Default False)", type=bool)
	parser.add_argument("-l","--loadfile", help="Load  model from saved checkpoint? (Default False)", type=bool)
	parser.add_argument("-rd","--rootdir", help="Root directory for project", type=str)
	parser.add_argument("-sd","--savedir", help="Save directory for project", type=str)
	parser.add_argument("-ld","--loaddir", help="Load directory for project", type=str)

	# Parse Arguments
	args = parser.parse_args()

	# Value Function Approximator Training
	num_training = args.trainsteps if args.trainsteps else 1000
	hidden_units = args.hidunits if args.hidunits else 100
	learning_rate = args.learnrate if args.learnrate else 0.001
	batch_size = args.batchsize if args.batchsize else 32

	# Simulation Parameters
	max_moves = args.maxmoves if args.maxmoves else 100
	epsilon = args.epsilon if args.epsilon else 0.2
	visualize = args.visualize if args.visualize else False
	print_moves = args.print if args.print else False
	algebraic = args.algebraic if args.algebraic else False

	# Load File
	load_file = args.loadfile if args.loadfile else False

	# File Paths
	dir_name = args.rootdir if args.rootdir else "/Users/jonathanzia"	# Root directory
	load_path = args.loaddir if args.loaddir else "checkpoints/model"	# Load previous model
	save_path = args.savedir if args.savedir else "checkpoints/model"	# Save model at each step

	filewriter_path = os.path.join(dir_name, "output")					# Filewriter save path
	training_loss = os.path.join(dir_name, "training_loss.txt")			# Training loss

	# ----------------------------------------------------
	# Train Model
	# ----------------------------------------------------
	
	"""
	Train value function model

	Arguments:
	- num_training:		[int]	Number of training steps
	- hidden_units:		[int]	Number of hidden units per layer
	- learning_rate:	[float]	Initial learning rate
	- batch_size:		[int]	Batch size for stochastic gradient descent
	- max_moves:		[int]	Maximum moves for Monte Carlo simulations
	- epsilon:			[float]	Epsilon-greedy policy parameter
	- visualize:		[bool]	Visualize game board during training?
	- print_moves:		[bool]	Print moves during training?
	- algebraic:		[bool]	Print moves using algebraic notation or long-form?
	- load_file:		[bool] 	Load pre-trained model?
	- dir_name:			[str]	Root directory filepath
	- load_path:		[str]	Path to pre-trained model from root directory
	- save_path:		[str]	Save path from root directory
	- filewriter_path:	[str]	Save path for filewriter (TensorBoard)
	- training_loss		[str]	Output .txt file name / path for training loss
	"""

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
	hidden1 = tf.contrib.layers.fully_connected(inputs,num_outputs=hidden_units)

	# Second fully-connected layer
	hidden2 = tf.contrib.layers.fully_connected(hidden1,num_outputs=hidden_units)

	# Output layer
	predictions = tf.contrib.layers.fully_connected(hidden2,num_outputs=1,activation_fn=None)


	# ----------------------------------------------------
	# Calculate Loss and Define Optimizer
	# ----------------------------------------------------
	# Calculating mean squared error of predictions and targets
	loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
	loss = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


	# ----------------------------------------------------
	# Run Session
	# ----------------------------------------------------
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()	# Instantiate Saver class
	t_loss = []	# Placeholder for training loss values
	with tf.Session() as sess:

		# Create Tensorboard graph
		writer = tf.summary.FileWriter(filewriter_path, sess.graph)
		#merged = tf.summary.merge_all()

		# If there is a model checkpoint saved, load the checkpoint. Else, initialize variables.
		if load_file:
			# Restore saved session
			saver.restore(sess, load_path)
		else:
			# Initialize the variables
			sess.run(init)

		# Obtain start time
		start_time = t.time()

		# For each training step, generate a random board
		for step in range(0,num_training):

			# Run game and generate feature and label batches
			features, labels = generate_game(batch_size=batch_size,max_moves=max_moves,epsilon=epsilon,visualize=visualize,print_move=print_moves,algebraic=algebraic)


			# ----------------------------------------------------
			# Optimize Model for Current Simulation
			# ----------------------------------------------------			
			# Print step
			print("\nOptimizing at step", step)
			# Run optimizer, loss, and predicted error ops in graph
			predictions_, targets_, _, loss_ = sess.run([predictions, targets, optimizer, loss],feed_dict={inputs: np.reshape(features,(batch_size,768)), targets: np.expand_dims(labels,axis=1)})

			# Record loss
			t_loss.append(loss_)

			# Save and overwrite the session at each training step
			saver.save(sess, save_path)
			# Writing summaries to Tensorboard at each training step
			#summ = sess.run(merged)
			#writer.add_summary(summ,step)

			# Conditional statement for calculating time remaining, percent completion, and loss
			if step % 10 == 0:

				# Report loss
				print("\nAverage train loss at step", step, ":", np.mean(t_loss))

				# Print predictions and targets
				#print("\nPredictions:")
				#print(predictions_)
				#print("\nTargets:")
				#print(targets_)

				# Report percent completion
				p_completion = 100*step/num_training
				print("\nPercent completion: %.3f%%" % p_completion)

				# Print time remaining
				avg_elapsed_time = (t.time() - start_time)/(step+1)
				sec_remaining = avg_elapsed_time*(num_training-step)
				min_remaining = round(sec_remaining/60)
				print("Time Remaining: %d minutes" % min_remaining)

		# Write training loss to file
		t_loss = np.array(t_loss)
		with open(training_loss, 'a') as file_object:
			np.savetxt(file_object, t_loss)

		# Close the writer
		writer.close()


