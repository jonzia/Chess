#---------------------------------------------
# Representing Board State for Chess AI
# Created By: Jonathan Zia
# Last Edited: Saturday, March 17 2018
# Georgia Institute of Technology
#---------------------------------------------
import numpy as np
import pieces as p
import random as r

def initialize_pieces(random=False, keep_prob=1.0):

	"""Construct list of pieces as objects"""

	# Args: (1) random: Whether board is initialized to random initial state
	#		(2) keep_prob: Probability of retaining piece
	# Returns: Python list of pieces
	# 1,1 = a1 ... 8,8 = h8

	piece_list = [p.Rook('white',1,1), p.Knight('white',2,1), p.Bishop('white',3,1), p.Queen('white'),
		p.King('white'), p.Bishop('white',6,1), p.Knight('white',7,1), p.Rook('white',8,1),
		p.Pawn('white',1,2), p.Pawn('white',2,2), p.Pawn('white',3,2), p.Pawn('white',4,2),
		p.Pawn('white',5,2), p.Pawn('white',6,2), p.Pawn('white',7,2), p.Pawn('white',8,2),
		p.Pawn('black',1,7), p.Pawn('black',2,7), p.Pawn('black',3,7), p.Pawn('black',4,7),
		p.Pawn('black',5,7), p.Pawn('black',6,7), p.Pawn('black',7,7), p.Pawn('black',8,7),
		p.Rook('black',1,8), p.Knight('black',2,8), p.Bishop('black',3,8), p.Queen('black'),
		p.King('black'), p.Bishop('black',6,8), p.Knight('black',7,8), p.Rook('black',8,8)]

	# If random is True, randomize piece positions and activity
	if random:
		# For piece in piece list...
		for piece in piece_list:
			# Toggle activity based on uniform distribution (AND PIECE IS NOT KING)
			if r.random() >= keep_prob and piece.name != 'King':
				piece.remove()
			# If the piece was not removed, randomize file and rank
			else:
				newfile = r.randint(1,8)
				newrank = r.randint(1,8)

				# If there is another piece in the target tile, swap places
				for other_piece in piece_list:
					if other_piece.is_active and other_piece.file == newfile and other_piece.rank == newrank:
						# Swap places
						other_piece.file = piece.file
						other_piece.rank = piece.rank
				# Else, and in the previous case, update the piece's file and rank
				piece.file = newfile
				piece.rank = newrank
				piece.move_count += 1


	return piece_list

def board_state(piece_list):

	"""Configuring inputs for value function network"""

	# Args: (1) piece list

	# The output contains M planes of dimensions (N X N) where (N X N) is the size of the board.
	# There are M planes "stacked" in layers where each layer represents a different "piece group" 
	# (e.g. white pawns, black rooks, etc.) in one-hot format where 1 represents a piece in those
	# coordinates and 0 represents the piece is not in those coordinates.

	# Define parameters
	N = 8	# N = board dimensions (8 x 8)
	M = 12	# M = piece groups (6 per player)

	# Initializing board state with dimensions N x N x (MT + L)
	board = np.zeros((N,N,M))

	# The M layers each represent a different piece group. The order of is as follows:
	# 0: White Pawns 		Pieces 8 - 15
	# 1: White Knights		Pieces 1 and 6
	# 2: White Bishops		Pieces 2 and 5
	# 3: White Rooks 		Pieces 0 and 7
	# 4: White Queen		Piece 3
	# 5: White King 		Piece 4
	# 6: Black Pawns 		Pieces 16 - 23
	# 7: Black Knights 		Pieces 25 and 30
	# 8: Black Bishops 		Pieces 26 and 29
	# 9: Black Rooks 		Pieces 24 and 31
	# 10: Black Queen		Piece 27
	# 11: Black King 		Piece 28
	# Note that the number of pieces in each category may change upon piece promotion or removal
	# (hence the code below will remain general).

	# Fill board state with pieces
	for piece in piece_list:
		# Place active white pawns in plane 0 and continue to next piece
		if piece.is_active and piece.color == 'white' and piece.name == 'Pawn':
			try:
				board[piece.file-1, piece.rank-1, 0] = 1
			except:
				continue
			continue
		# Place active white knights in plane 1 and continue to next piece
		elif piece.is_active and piece.color == 'white' and piece.name == 'Knight':
			try:
				board[piece.file-1, piece.rank-1, 1] = 1
			except:
				continue
			continue
		# Place active white bishops in plane 2 and continue to next piece
		elif piece.is_active and piece.color == 'white' and piece.name == 'Bishop':
			try:
				board[piece.file-1, piece.rank-1, 2] = 1
			except:
				continue
			continue
		# Place active white rooks in plane 3 and continue to next piece
		elif piece.is_active and piece.color == 'white' and piece.name == 'Rook':
			try:
				board[piece.file-1, piece.rank-1, 3] = 1
			except:
				continue
			continue
		# Place active white queen(s) in plane 4 and continue to next piece
		elif piece.is_active and piece.color == 'white' and piece.name == 'Queen':
			try:
				board[piece.file-1, piece.rank-1, 4] = 1
			except:
				continue
			continue
		# Place active white king in plane 5 and continue to next piece
		elif piece.is_active and piece.color == 'white' and piece.name == 'King':
			try:
				board[piece.file-1, piece.rank-1, 5] = 1
			except:
				continue
			continue
		# Place active black pawns in plane 6 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'Pawn':
			try:
				board[piece.file-1, piece.rank-1, 6] = 1
			except:
				continue
			continue
		# Place active black knights in plane 7 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'Knight':
			try:
				board[piece.file-1, piece.rank-1, 7] = 1
			except:
				continue
			continue
		# Place active black bishops in plane 8 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'Bishop':
			try:
				board[piece.file-1, piece.rank-1, 8] = 1
			except:
				continue
			continue
		# Place active black rooks in plane 9 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'Rook':
			try:
				board[piece.file-1, piece.rank-1, 9] = 1
			except:
				continue
			continue
		# Place active black queen(s) in plane 10 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'Queen':
			try:
				board[piece.file-1, piece.rank-1, 10] = 1
			except:
				continue
			continue
		# Place active black king in plane 11 and continue to next piece
		elif piece.is_active and piece.color == 'black' and piece.name == 'King':
			try:
				board[piece.file-1, piece.rank-1, 11] = 1
			except:
				continue

	# Return board state
	return board

def visualize_state(piece_list):

	"""Visualizing board in terminal"""

	# Args: (1) piece list

	# The output is an 8x8 grid indicating the present locations for each piece

	# Initializing empty grid
	visualization = np.empty([8,8],dtype=object)
	for i in range(0,8):
		for j in range(0,8):
			visualization[i,j] = ' ';

	for piece in piece_list:
		# Load active pawns
		if piece.is_active and piece.color == 'white' and piece.name == 'Pawn':
			visualization[piece.file-1, piece.rank-1] = 'P'
		elif piece.is_active and piece.color == 'black' and piece.name == 'Pawn':
			visualization[piece.file-1, piece.rank-1] = 'p'
		elif piece.is_active and piece.color == 'white' and piece.name == 'Rook':
			visualization[piece.file-1, piece.rank-1] = 'R'
		elif piece.is_active and piece.color == 'black' and piece.name == 'Rook':
			visualization[piece.file-1, piece.rank-1] = 'r'
		elif piece.is_active and piece.color == 'white' and piece.name == 'Knight':
			visualization[piece.file-1, piece.rank-1] = 'N'
		elif piece.is_active and piece.color == 'black' and piece.name == 'Knight':
			visualization[piece.file-1, piece.rank-1] = 'n'
		elif piece.is_active and piece.color == 'white' and piece.name == 'Bishop':
			visualization[piece.file-1, piece.rank-1] = 'B'
		elif piece.is_active and piece.color == 'black' and piece.name == 'Bishop':
			visualization[piece.file-1, piece.rank-1] = 'b'
		elif piece.is_active and piece.color == 'white' and piece.name == 'Queen':
			visualization[piece.file-1, piece.rank-1] = 'Q'
		elif piece.is_active and piece.color == 'black' and piece.name == 'Queen':
			visualization[piece.file-1, piece.rank-1] = 'q'
		elif piece.is_active and piece.color == 'white' and piece.name == 'King':
			visualization[piece.file-1, piece.rank-1] = 'K'
		elif piece.is_active and piece.color == 'black' and piece.name == 'King':
			visualization[piece.file-1, piece.rank-1] = 'k'

	# Return visualization
	return visualization


def action_space(piece_list, player):

	"""Determining available moves for evaluation"""

	# Args: (1) piece list, (2) player color

	# The output is a P x 56 matrix where P is the number of pieces and 56 is the maximum
	# possible number of moves for any piece. For pieces which have less than  possible
	# moves, zeros are appended to the end of the row. A value of 1 indicates that a
	# move is available while a value of 0 means that it is not.


	# See pieces.py for move glossary

	# Initializing action space with dimensions P x 56
	action_space = np.zeros((16,56))

	# For each piece...
	for i in range(0,16):
		# If it is white's turn to move...
		if player == 'white':
			# Obtain vector of possible actions and write to corresponding row
			action_space[i,:] = piece_list[i].actions(piece_list)
		else:
			action_space[i,:] = piece_list[i+16].actions(piece_list)

	# Return action space
	return action_space

def points(piece_list):

	"""Calculating point differential for the given board state"""

	# Args: (1) piece list
	# Returns: differential (white points - black points)

	# The points are calculated via the standard chess value system:
	# Pawn = 1, King = 3, Bishop = 3, Rook = 5, Queen = 9
	# King = 100 (arbitrarily large)

	differential = 0
	# For all white pieces...
	for i in range(0,16):
		# If the piece is active, add its points to the counter
		if piece_list[i].is_active:
			differential = differential + piece_list[i].value
	# For all black pieces...
	for i in range(16,32):
		# If the piece is active, subtract its points from the counter
		if piece_list[i].is_active:
			differential = differential - piece_list[i].value

	# Return point differential
	return differential
