# Python3 solution for 
# the problem Grid Construction
# in the PRACTICE section of HackerEarth

# Utility function to print
# the grid
def print_grid(arr):
	for i in range(n):
		for j in range(n):
			print(arr[i][j], end="")
		print()
			
# Function to construct the grid
def construct_grid():
	# Create an empty grid
	arr = [[0 for i in range(n)]
			for j in range(n)]
			
	# Fill the grid
	for i in range(n):
		for j in range(n):
			# If the row is before
			# the column in the
			# permutation P, then
			# the cell should be 0
			if (permP[i] < permQ[j]):
				arr[i][j] = 0
			# Otherwise, the cell
			# should be 1
			else:
				arr[i][j] = 1
				
	# Print the grid
	print_grid(arr)
				
# Driver code
n = int(input())
permP = list(map(int, input().split()))
permQ = list(map(int, input().split()))

# Adjust the permutations
# to be 0-indexed
for i in range(n):
	permP[i] -= 1
	permQ[i] -= 1

# Construct and print the grid
construct_grid()