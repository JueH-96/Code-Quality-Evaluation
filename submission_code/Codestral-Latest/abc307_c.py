# YOUR CODE HERE
import sys
from itertools import product

def can_achieve_goal(A, B, X):
    H_A, W_A = len(A), len(A[0])
    H_B, W_B = len(B), len(B[0])
    H_X, W_X = len(X), len(X[0])

    # Find all black squares in A and B
    black_squares_A = [(i, j) for i, j in product(range(H_A), range(W_A)) if A[i][j] == '#']
    black_squares_B = [(i, j) for i, j in product(range(H_B), range(W_B)) if B[i][j] == '#']

    # Check all possible positions to paste A and B
    for pos_A in product(range(-H_X + 1, H_A), range(-W_X + 1, W_A)):
        for pos_B in product(range(-H_X + 1, H_B), range(-W_X + 1, W_B)):
            # Create a grid for the cut-out sheet
            cut_out = [['.' for _ in range(W_X)] for _ in range(H_X)]

            # Paste A onto the cut-out sheet
            for i, j in black_squares_A:
                x, y = i + pos_A[0], j + pos_A[1]
                if 0 <= x < H_X and 0 <= y < W_X:
                    cut_out[x][y] = '#'

            # Paste B onto the cut-out sheet
            for i, j in black_squares_B:
                x, y = i + pos_B[0], j + pos_B[1]
                if 0 <= x < H_X and 0 <= y < W_X:
                    cut_out[x][y] = '#'

            # Check if the cut-out sheet matches X
            if cut_out == X:
                return True

    return False

# Read input
H_A, W_A = map(int, input().split())
A = [input().strip() for _ in range(H_A)]
H_B, W_B = map(int, input().split())
B = [input().strip() for _ in range(H_B)]
H_X, W_X = map(int, input().split())
X = [input().strip() for _ in range(H_X)]

# Determine if Takahashi can achieve the goal
result = can_achieve_goal(A, B, X)
print("Yes" if result else "No")