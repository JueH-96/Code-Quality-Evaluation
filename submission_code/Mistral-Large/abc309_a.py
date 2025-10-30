import sys

# Read input from stdin
input = sys.stdin.read()
A, B = map(int, input.split())

# Define the 3x3 board
board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Function to get the position of a number on the board
def get_position(num):
    for i in range(3):
        for j in range(3):
            if board[i][j] == num:
                return (i, j)
    return None

# Get positions of A and B
pos_A = get_position(A)
pos_B = get_position(B)

# Check if A and B are adjacent horizontally
if pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
    print("Yes")
else:
    print("No")