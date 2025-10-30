A, B = map(int, input().split())

# Define the 3x3 board as a 2D list
board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Find positions of A and B
pos_A = None
pos_B = None

for i in range(3):
    for j in range(3):
        if board[i][j] == A:
            pos_A = (i, j)
        if board[i][j] == B:
            pos_B = (i, j)

# Check if they are horizontally adjacent
# They must be in same row and columns must differ by 1
if pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
    print("Yes")
else:
    print("No")