# YOUR CODE HERE
def are_adjacent_horizontally(A, B):
    # Define the board as a list of lists
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
    if pos_A and pos_B:
        if pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
            return "Yes"
    
    return "No"

# Read input
import sys
input = sys.stdin.read
A, B = map(int, input().strip().split())

# Print the result
print(are_adjacent_horizontally(A, B))