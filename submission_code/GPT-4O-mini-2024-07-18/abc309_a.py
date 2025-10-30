# YOUR CODE HERE
def are_adjacent_horizontally(A, B):
    # Define the 3x3 board
    board = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    # Create a position map for quick lookup
    position = {}
    for i in range(3):
        for j in range(3):
            position[board[i][j]] = (i, j)
    
    # Get the positions of A and B
    pos_A = position[A]
    pos_B = position[B]
    
    # Check if they are in the same row and adjacent
    if pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input_data = sys.stdin.read().strip()
A, B = map(int, input_data.split())

# Output the result
print(are_adjacent_horizontally(A, B))