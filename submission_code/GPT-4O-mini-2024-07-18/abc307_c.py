def can_achieve_goal(A, B, X):
    H_A, W_A = len(A), len(A[0])
    H_B, W_B = len(B), len(B[0])
    H_X, W_X = len(X), len(X[0])
    
    # Collect all black squares from A and B
    black_squares = set()
    
    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                black_squares.add((i, j))
    
    for i in range(H_B):
        for j in range(W_B):
            if B[i][j] == '#':
                black_squares.add((i, j))
    
    # Check all possible placements of X on the combined area of A and B
    for start_row in range(-H_A, H_X + 1):
        for start_col in range(-W_A, W_X + 1):
            # Check if we can cut out X from the combined area
            can_cut = True
            for i in range(H_X):
                for j in range(W_X):
                    if X[i][j] == '#':
                        # Calculate the corresponding position in the combined area
                        combined_row = start_row + i
                        combined_col = start_col + j
                        if not (0 <= combined_row < H_A + H_B and 0 <= combined_col < W_A + W_B):
                            can_cut = False
                            break
                        if (combined_row < H_A and combined_col < W_A and A[combined_row][combined_col] == '.') and \
                           (combined_row - H_A < H_B and combined_col < W_B and B[combined_row - H_A][combined_col] == '.'):
                            can_cut = False
                            break
                if not can_cut:
                    break
            
            if can_cut:
                # Now check if all black squares from A and B are included
                for (i, j) in black_squares:
                    if not (start_row <= i < start_row + H_X and start_col <= j < start_col + W_X):
                        can_cut = False
                        break
            
            if can_cut:
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().splitlines()

# Read sheets A, B, and X
H_A, W_A = map(int, data[0].split())
A = data[1:H_A + 1]
H_B, W_B = map(int, data[H_A + 1].split())
B = data[H_A + 2:H_A + 2 + H_B]
H_X, W_X = map(int, data[H_A + 2 + H_B].split())
X = data[H_A + 3 + H_B:H_A + 3 + H_B + H_X]

# Determine if the goal can be achieved
result = can_achieve_goal(A, B, X)
print(result)