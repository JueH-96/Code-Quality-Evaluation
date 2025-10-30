def can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X):
    # Iterate over all possible positions to place A on C
    for xA in range(-H_A + 1, H_X):
        for yA in range(-W_A + 1, W_X):
            # Iterate over all possible positions to place B on C
            for xB in range(-H_B + 1, H_X):
                for yB in range(-W_B + 1, W_X):
                    # Create a blank sheet C of size H_X x W_X
                    C = [['.'] * W_X for _ in range(H_X)]
                    
                    # Place A on C
                    for i in range(H_A):
                        for j in range(W_A):
                            if A[i][j] == '#':
                                x = xA + i
                                y = yA + j
                                if 0 <= x < H_X and 0 <= y < W_X:
                                    C[x][y] = '#'
                    
                    # Place B on C
                    for i in range(H_B):
                        for j in range(W_B):
                            if B[i][j] == '#':
                                x = xB + i
                                y = yB + j
                                if 0 <= x < H_X and 0 <= y < W_X:
                                    C[x][y] = '#'
                    
                    # Check if C matches X
                    match = True
                    for i in range(H_X):
                        for j in range(W_X):
                            if C[i][j] != X[i][j]:
                                match = False
                                break
                        if not match:
                            break
                    
                    if match:
                        return "Yes"
    
    return "No"

# Read inputs
import sys
input = sys.stdin.read
data = input().split()

# Parse inputs
index = 0
H_A = int(data[index])
W_A = int(data[index + 1])
index += 2
A = data[index:index + H_A]
index += H_A

H_B = int(data[index])
W_B = int(data[index + 1])
index += 2
B = data[index:index + H_B]
index += H_B

H_X = int(data[index])
W_X = int(data[index + 1])
index += 2
X = data[index:index + H_X]

# Solve the problem
result = can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X)
print(result)