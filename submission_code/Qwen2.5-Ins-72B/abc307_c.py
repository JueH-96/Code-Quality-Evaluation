def can_form_sheet(A, B, X):
    H_A, W_A = len(A), len(A[0])
    H_B, W_B = len(B), len(B[0])
    H_X, W_X = len(X), len(X[0])
    
    # Function to check if a given position (dx, dy) for A and B can form X
    def check(dx, dy):
        for i in range(H_X):
            for j in range(W_X):
                if X[i][j] == '#':
                    if (0 <= i - dx < H_A and 0 <= j - dy < W_A and A[i - dx][j - dy] == '#') or \
                       (0 <= i - dx < H_B and 0 <= j - dy < W_B and B[i - dx][j - dy] == '#'):
                        continue
                    return False
        return True
    
    # Check all possible positions for A and B
    for dx in range(H_X - H_A + 1):
        for dy in range(W_X - W_A + 1):
            if check(dx, dy):
                return True
    
    for dx in range(H_X - H_B + 1):
        for dy in range(W_X - W_B + 1):
            if check(dx, dy):
                return True
    
    return False

# Read input
H_A, W_A = map(int, input().split())
A = [input().strip() for _ in range(H_A)]
H_B, W_B = map(int, input().split())
B = [input().strip() for _ in range(H_B)]
H_X, W_X = map(int, input().split())
X = [input().strip() for _ in range(H_X)]

# Solve the problem
if can_form_sheet(A, B, X):
    print("Yes")
else:
    print("No")