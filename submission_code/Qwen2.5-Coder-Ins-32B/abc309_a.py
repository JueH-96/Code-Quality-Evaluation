# YOUR CODE HERE
import sys

def are_adjacent_horizontally(A, B):
    # Define the 3x3 board
    board = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    # Find the positions of A and B
    pos_A = None
    pos_B = None
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == A:
                pos_A = (i, j)
            if board[i][j] == B:
                pos_B = (i, j)
    
    # Check if they are adjacent horizontally
    if pos_A and pos_B and pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
        return "Yes"
    else:
        return "No"

# Read input
input = sys.stdin.read().strip().split()
A = int(input[0])
B = int(input[1])

# Output the result
print(are_adjacent_horizontally(A, B))