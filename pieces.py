#---------------------------------------------
# Chess Piece Classes for Chess AI
# Created By: Jonathan Zia
# Last Edited: Saturday, March 17 2018
# Georgia Institute of Technology
#---------------------------------------------
import tensorflow as tf
import numpy as np
import pieces as p
import random as r
import state as s
import time as t
import copy as c
import math
import os


#---------------------------------------------
# Pawn Class
#---------------------------------------------
class Pawn():

	"""Defining attributes of pawn piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Pawn'		# Name
		self.value = 1			# Value (1 for pawn)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting Position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current Position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires:	piece_list
		# Returns:	numpy array

		# The pawn may move up to two spaces forward on its first move
		# and one space forward on each subsequent move. It may move 
		# one space in the forward diagonal direction if attacking an
		# enemy piece.

		# IF THE PIECE IS A PAWN (HAS NOT BEEN PROMOTED)
		if self.name == 'Pawn':

			# For each vector in movement array:
			# (1) For forward movement, there is no piece in the path and index is in bounds
			# (2) For attack movement, there must be a piece of a different color in path
			# (3) For moving forward 2 spaces, move_count must equal 0

			# Initialize action vector:
			# [1 forward, 2 forward, attack (+file), attack (-file), promotion, 51 zeros]
			action_space = np.zeros((1,56))

			# Initialize coordinate vector
			coordinates = []

			if self.is_active:

				if self.color == 'white':
					# Initialize movement vector array (file, rank)
					movement = np.array([[0,1],[0,2],[1,1],[-1,1]])

					for i in range(0,4):
							# Condition (1)
							if i == 0 and 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
								blocked = False
								for piece in piece_list:
									if piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
									action_space[0,i] = 1
							# Condition (2)
							if i == 2 or i == 3:
								if 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
									for piece in piece_list:
										if piece.is_active and piece.color != self.color and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
											coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
											action_space[0,i] = 1
											break
							# Condition (3)
							if i == 1 and self.move_count == 0:
								for piece in piece_list:
									blocked = False
									if piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
										blocked = True
										break
									elif piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]-1:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
									action_space[0,i] = 1
				if self.color == 'black':
					# Initialize movement vector array (file, rank)
					movement = np.array([[0,-1],[0,-2],[1,-1],[-1,-1]])

					for i in range(0,4):
							# Condition (1)
							if i == 0 and 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
								for piece in piece_list:
									blocked = False
									if piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
									action_space[0,i] = 1
							# Condition (2)
							if i == 2 or i == 3:
								if 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
									for piece in piece_list:
										if piece.is_active and piece.color != self.color and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
											coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
											action_space[0,i] = 1
											break
							# Condition (3)
							if i == 1 and self.move_count == 0:
								for piece in piece_list:
									blocked = False
									if piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
										blocked = True
										break
									elif piece.is_active and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]+1:
										blocked = True
										break
								if blocked == False:
									coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
									action_space[0,i] = 1

				# Can pawn promote to queen?
				Promote = False
				# If the pawn is white and has rank 8 or is black and has rank 1, it can promote to queen
				if self.color == 'white' and self.rank == 8:
					Promote = True
				elif self.color == 'black' and self.rank == 1:
					Promote = True
				# If AttackRight is True, append special coordinates
				if Promote:
					coordinates.append([0, 0])
					action_space[0,4] = 1

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space


		# IF THE PIECE IS A QUEEN (HAS BEEN PROMOTED)
		else:

			# The queen's movement is a combination of bishop and rook

			# VERTICAL/HORIZONTAL
			# For each tile along one of the four movement vectors, append coordinate if:
			# (1) The index is in bounds
			# (2) There is no piece of the same color
			# (3) There was no piece of the opposite color in the preceding step

			# Initialize action vector:
			# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]
			action_space = np.zeros((1,56))

			# Initialize coordinate aray
			coordinates = []

			if self.is_active:

				# Initialize movement vector array (file, rank)
				movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

				for i in range(0,4):
					break_loop = False
					for j in range(1,8):
						# Condition (1)
						if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
							for piece in piece_list:
								# Condition 2
								if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
									break_loop = True
								# Condition 3
								if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
									break_loop = True
						else: # If the index is no longer in bounds, break
							break
						if break_loop: # If the break_loop was thrown, break
							break
						# If the break_loop was not thrown, append coordinates
						coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
						action_space[0,7*i+(j-1)] = 1

				# DIAGONAL
				# For each tile along one of the four movement vectors, append coordinate if:
				# (1) The index is in bounds
				# (2) There is no piece of the same color
				# (3) There was no piece of the opposite color in the preceding step

				# Initialize movement vector array (file, rank)
				movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

				for i in range(0,4):
					break_loop = False
					for j in range(1,8):
						# Condition (1)
						if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
							for piece in piece_list:
								# Condition 2
								if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
									break_loop = True
								# Condition 3
								if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
									break_loop = True
						else: # If the index is no longer in bounds, break
							break
						if break_loop: # If the break_loop was thrown, break
							break
						# If the break_loop was not thrown, append coordinates
						coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
						action_space[0,7*i+(j-1)+28] = 1

				# Convert coordinates to numpy array
				coordinates = np.asarray(coordinates)

			# Return possible moves
			if return_coordinates:
				return coordinates
			else:
				return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# IF THE PIECE IS A PAWN (HAS NOT BEEN PROMOTED)
		if self.name == "Pawn":

			# Action vector:
			# [1 forward, 2 forward, attack (+file), attack (-file), promotion, 51 zeros]
			
			# Move 1 forward
			if action == 0:
				if self.color == 'white':
					self.rank = self.rank+1
				else:
					self.rank = self.rank-1
			# Move 2 forward
			elif action == 1:
				if self.color == 'white':
					self.rank = self.rank+2
				else:
					self.rank = self.rank-2
			# Attack (+file)
			elif action == 2:
				if self.color == 'white':
					self.file = self.file+1
					self.rank = self.rank+1
				else:
					self.file = self.file+1
					self.rank = self.rank-1
			# Attack (-file)
			elif action == 3:
				if self.color == 'white':
					self.file = self.file-1
					self.rank = self.rank+1
				else:
					self.file = self.file-1
					self.rank = self.rank-1
			# Promote to queen
			else:
				self.name = 'Queen'
				self.value = 9


		# IF THE PIECE IS A QUEEN (HAS BEEN PROMOTED)
		else:

			# Action vector:
			# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]

			# +file movements
			if 0 <= action < 7:
				self.file = self.file + (action+1)
			# -file movements
			elif 7 <= action < 14:
				self.file = self.file - (action-6)
			# +rank movements
			elif 14 <= action < 21:
				self.rank = self.rank + (action-13)
			# -rank movements
			elif 21 <= action < 28:
				self.rank = self.rank - (action-20)
			# +f/+r movements
			elif 28 <= action < 35:
				self.file = self.file + (action-27)
				self.rank = self.rank + (action-27)
			# +f/-r movements
			elif 35 <= action < 42:
				self.file = self.file + (action-34)
				self.rank = self.rank - (action-34)
			# -f/+r movements
			elif 42 <= action < 49:
				self.file = self.file - (action-41)
				self.rank = self.rank + (action-41)
			# -f/-r movements
			else:
				self.file = self.file - (action-48)
				self.rank = self.rank - (action-48)


		# Increment move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False

#---------------------------------------------
# Rook Class
#---------------------------------------------
class Rook():

	"""Defining attributes of rook piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Rook'		# Name
		self.value = 5			# Value (5 for rook)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""	

		# Requires: piece_list
		# Returns: numpy array

		# The rook may move any number of spaces along its current rank/file.
		# It may also attack opposing pieces in its movement path.

		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [1-7 +file, 1-7 -file, 1-7 +rank, 1-7 -rank, 28 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)] = 1

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# Action vector:
		# [1-7 +file, 1-7 -file, 1-7 +rank, 1-7 -rank, 28 zeros]

		# +file movements
		if 0 <= action < 7:
			self.file = self.file + (action+1)
		# -file movements
		elif 7 <= action < 14:
			self.file = self.file - (action-6)
		# +rank movements
		elif 14 <= action < 21:
			self.rank = self.rank + (action-13)
		# -rank movements
		else:
			self.rank = self.rank - (action-20)


		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False


#---------------------------------------------
# Knight Class
#---------------------------------------------
class Knight():

	"""Defining attributes of knight piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Knight'	# Name
		self.value = 3			# Value (3 for knight)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# A knight may have any of 8 possible actions:
		# Move forward 2 tiles in any direction + 1 tile perpendicularly

		# For each of the 8 possible actions, if:
		# (1) The index is not out of bounds
		# (2) There is not a piece of the same color
		# Then append the coordinates to the output array

		# Initialize action vector:
		# [8 knight moves, 48 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement array (file, rank)
			movement = np.array([[2,1],[2,-1],[1,2],[-1,2],[1,-2],[-1,-2],[-2,1],[-2,-1]])
			for i in range(0,8):
				continue_loop = False
				# Condition (1)
				if 0 < self.file+movement[i,0] < 9 and 0 < self.rank+movement[i,1] < 9:
					for piece in piece_list:
						# Condition (2)
						if piece.is_active and piece.color == self.color and piece.file == self.file+movement[i,0] and piece.rank == self.rank+movement[i,1]:
							continue_loop = True
							break
				else: # If the index is not in bounds, continue
					continue
				# If continue_loop is True, continue the loop without appending coordinate. Else, append coordinate.
				if continue_loop:
					continue
				coordinates.append([self.file+movement[i,0], self.rank+movement[i,1]])
				action_space[0,i] = 1


			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# Action vector:
		# [[2,1],[2,-1],[1,2],[-1,2],[1,-2],[-1,-2],[-2,1],[-2,-1]]

		if action == 0:
			self.file = self.file + 2
			self.rank = self.rank + 1
		elif action == 1:
			self.file = self.file + 2
			self.rank = self.rank - 1
		elif action == 2:
			self.file = self.file + 1
			self.rank = self.rank + 2
		elif action == 3:
			self.file = self.file - 1
			self.rank = self.rank + 2
		elif action == 4:
			self.file = self.file + 1
			self.rank = self.rank - 2
		elif action == 5:
			self.file = self.file - 1
			self.rank = self.rank - 2
		elif action == 6:
			self.file = self.file - 2
			self.rank = self.rank + 1
		else:
			self.file = self.file - 2
			self.rank = self.rank - 1

		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False


#---------------------------------------------
# Bishiop Class
#---------------------------------------------
class Bishop():

	"""Defining attributes of bishop piece"""

	def __init__(self, color, start_file, start_rank):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Bishop'	# Name
		self.value = 3			# Value (3 for bishop)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		self.start_file = start_file
		self.start_rank = start_rank
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = start_file
		self.rank = start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# The bishop can move diagonally in any direction.

		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r, 28 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)] = 1

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# Action vector:
		# [1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r, 28 zeros]

		# +f/+r movements
		if 0 <= action < 7:
			self.file = self.file + (action+1)
			self.rank = self.rank + (action+1)
		# +f/-r movements
		elif 7 <= action < 14:
			self.file = self.file + (action-6)
			self.rank = self.rank - (action-6)
		# -f/+r movements
		elif 14 <= action < 21:
			self.file = self.file - (action-13)
			self.rank = self.rank + (action-13)
		# -f/-r movements
		else:
			self.file = self.file - (action-20)
			self.rank = self.rank - (action-20)

		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False


#---------------------------------------------
# Queen Class
#---------------------------------------------
class Queen():

	"""Defining attributes of queen piece"""

	def __init__(self, color):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'Queen'		# Name
		self.value = 9			# Value (9 for queen)
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		if color == 'white':
			self.start_file = 4
			self.start_rank = 1
		else:
			self.start_file = 4
			self.start_rank = 8
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = self.start_file
		self.rank = self.start_rank


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# The queen's movement is a combination of bishop and rook

		# VERTICAL/HORIZONTAL
		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color
		# (3) There was no piece of the opposite color in the preceding step

		# Initialize action vector:
		# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)] = 1

			# DIAGONAL
			# For each tile along one of the four movement vectors, append coordinate if:
			# (1) The index is in bounds
			# (2) There is no piece of the same color
			# (3) There was no piece of the opposite color in the preceding step

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,8):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
							# Condition 3
							if piece.is_active and piece.color != self.color and piece.file == self.file+(j-1)*movement[i,0] and piece.rank == self.rank+(j-1)*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,7*i+(j-1)+28] = 1

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# Action vector:
		# [1-7 +f, 1-7 -f, 1-7 +r, 1-7 -r, 1-7 +f/+r, 1-7 +f/-r, 1-7 -f/+r, 1-7 -f/-r]

		# +file movements
		if 0 <= action < 7:
			self.file = self.file + (action+1)
		# -file movements
		elif 7 <= action < 14:
			self.file = self.file - (action-6)
		# +rank movements
		elif 14 <= action < 21:
			self.rank = self.rank + (action-13)
		# -rank movements
		elif 21 <= action < 28:
			self.rank = self.rank - (action-20)
		# +f/+r movements
		elif 28 <= action < 35:
			self.file = self.file + (action-27)
			self.rank = self.rank + (action-27)
		# +f/-r movements
		elif 35 <= action < 42:
			self.file = self.file + (action-34)
			self.rank = self.rank - (action-34)
		# -f/+r movements
		elif 42 <= action < 49:
			self.file = self.file - (action-41)
			self.rank = self.rank + (action-41)
		# -f/-r movements
		else:
			self.file = self.file - (action-48)
			self.rank = self.rank - (action-48)


		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False


#---------------------------------------------
# King Class
#---------------------------------------------
class King():

	"""Defining attributes of king piece"""

	def __init__(self, color):

		"""Defining initial attributes of piece"""

		# Piece Attributes
		self.name = 'King'		# Name
		self.value = 100		# Value
		self.color = color		# Color
		self.is_active = True	# Active/Inactive

		# Starting position
		# File = vertical column (a = 1 = queenside ... h = 8 = kingside)
		# Rank = horizontal row (1 = white ... 8 = black)
		if color == 'white':
			self.start_file = 5
			self.start_rank = 1
		else:
			self.start_file = 5
			self.start_rank = 8
		# Initializing move counter (increment when moved)
		self.move_count = 0

		# Current position
		self.file = self.start_file
		self.rank = self.start_rank

		# Special attributes
		# Can kingside castle?
		self.kCastle = False
		# Can queenside castle?
		self.qCastle = False


	# Returning numpy array with possible actions for piece
	# Array format:
	# [[file1 rank1]
	#  [file2 rank2]...]
	def actions(self, piece_list, return_coordinates=False):

		"""Determining possible actions for piece"""

		# Requires: piece_list
		# Returns: numpy array

		# The king may move one tile in any direction. The king may castle as a first move.
		# Special case of the queen where "j" is fixed to 1.

		# VERTICAL/HORIZONTAL
		# For each tile along one of the four movement vectors, append coordinate if:
		# (1) The index is in bounds
		# (2) There is no piece of the same color

		# Initialize action vector:
		# [8 king moves, kingside castle, queenside castle, 46 zeros]
		action_space = np.zeros((1,56))

		# Initialize coordinate aray
		coordinates = []

		if self.is_active:

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,0],[-1,0],[0,1],[0,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,2):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,i] = 1

			# DIAGONAL
			# For each tile along one of the four movement vectors, append coordinate if:
			# (1) The index is in bounds
			# (2) There is no piece of the same color

			# Initialize movement vector array (file, rank)
			movement = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

			for i in range(0,4):
				break_loop = False
				for j in range(1,2):
					# Condition (1)
					if 0 < self.file+j*movement[i,0] < 9 and 0 < self.rank+j*movement[i,1] < 9:
						for piece in piece_list:
							# Condition 2
							if piece.is_active and piece.color == self.color and piece.file == self.file+j*movement[i,0] and piece.rank == self.rank+j*movement[i,1]:
								break_loop = True
					else: # If the index is no longer in bounds, break
						break
					if break_loop: # If the break_loop was thrown, break
						break
					# If the break_loop was not thrown, append coordinates
					coordinates.append([self.file+j*movement[i,0], self.rank+j*movement[i,1]])
					action_space[0,i+4] = 1


			# Can king perform kingside castle?
			Castle = True

			# Conditions:
			# (1) If the king and kingside rook have not moved
			# (2) There are no pieces in between them
			# (3) The king is not in check

			# Condition (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[7].move_count == 0) or (self.color == 'black' and piece_list[31].move_count == 0)):
				for piece in piece_list:
					# Condition (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file+1 or piece.file == self.file+2):
						Castle = False
						break
					# Condition (3)
					elif piece.is_active and piece.name != 'King' and piece.color != self.color and piece.actions(piece_list, True).size > 0 and (piece.actions(piece_list, True) == np.array([self.file, self.rank])).all(1).any():
						Castle = False
						break
					else:
						Castle = True
			else:
				Castle = False

			if Castle:
				self.kCastle = True
				coordinates.append([0, 0])
				action_space[0,8] = 1
			else:
				self.kCastle = False

			# Can king perform queenside castle?
			Castle = True

			# Conditions:
			# (1) If the king and queenside rook have not moved
			# (2) There are no pieces in between them
			# (3) The king is not in check

			# Condition (1)
			if self.move_count == 0 and ((self.color == 'white' and piece_list[0].move_count == 0) or (self.color == 'black' and piece_list[24].move_count == 0)):
				for piece in piece_list:
					# Condition (2)
					if piece.is_active and piece.rank == self.rank and (piece.file == self.file-1 or piece.file == self.file-2 or piece.file == self.file-3):
						Castle = False
						break
					# Condition (3)
					elif piece.is_active and piece.name != 'King' and piece.color != self.color and piece.actions(piece_list, True).size > 0 and (piece.actions(piece_list, True) == np.array([self.file, self.rank])).all(1).any():
						Castle = False
						break
					else:
						Castle = True
			else:
				Castle = False

			if Castle:
				self.qCastle = True
				coordinates.append([-1,-1])
				action_space[0,9] = 1
			else:
				self.qCastle = False

			# Convert coordinates to numpy array
			coordinates = np.asarray(coordinates)

		# Return possible moves
		if return_coordinates:
			return coordinates
		else:
			return action_space


	def move(self, action, piece_list, print_move=False):

		"""Moving piece's position"""

		# Requires:	action (element of action vector)
		# Returns:	void

		# Action vector:
		# [8 king moves, kingside castle, queenside castle, 46 zeros]
		# [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1], kC, qC]

		if action == 0:
			self.file = self.file + 1
		elif action == 1:
			self.file = self.file - 1
		elif action == 2:
			self.rank = self.rank + 1
		elif action == 3:
			self.rank = self.rank - 1
		elif action == 4:
			self.file = self.file + 1
			self.rank = self.rank + 1
		elif action == 5:
			self.file = self.file + 1
			self.rank = self.rank - 1
		elif action == 6:
			self.file = self.file - 1
			self.rank = self.rank + 1
		elif action == 7:
			self.file = self.file - 1
			self.rank = self.rank - 1
		# Kingside castle
		elif action == 8:
			self.file = self.file + 2
			if self.color == 'white':
				piece_list[7].file = piece_list[7].file - 2
			else:
				piece_list[31].file = piece_list[31].file - 2
		# Queenside castle
		else:
			self.file = self.file - 2
			if self.color == 'white':
				piece_list[0].file = piece_list[0].file + 3
			else:
				piece_list[24].file = piece_list[24].file + 3

		# Update move counter
		self.move_count += 1

		# If a piece was in the destination tile, remove the piece
		piece_remove = False
		for piece in piece_list:
			if piece.is_active and piece.color != self.color and piece.file == self.file and piece.rank == self.rank:
				piece.remove()
				piece_remove = True
				remove_name = piece.name
				break

		# Print movement if indicated
		if print_move:
			if piece_remove:
				print(self.name + " to " + str(self.file) + "," + str(self.rank) + " taking " + remove_name)
			else:
				print(self.name + " to " + str(self.file) + "," + str(self.rank))


	def remove(self):

		"""Removing piece from board"""

		# Requires:	none
		# Returns:	void
		self.is_active = False