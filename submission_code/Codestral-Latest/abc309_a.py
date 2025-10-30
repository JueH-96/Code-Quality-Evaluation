# YOUR CODE HERE
import sys

# Read input
A, B = map(int, sys.stdin.read().split())

# Define the 3x3 board
board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Function to check if A and B are adjacent horizontally
def are_adjacent_horizontally(A, B):
    for row in board:
        for i in range(len(row) - 1):
            if (row[i] == A and row[i + 1] == B) or (row[i] == B and row[i + 1] == A):
                return True
    return False

# Determine if A and B are adjacent horizontally
if are_adjacent_horizontally(A, B):
    print("Yes")
else:
    print("No")