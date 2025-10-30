# Read input dimensions
N, M = map(int, input().split())

# Read grid S
S = []
for _ in range(N):
    row = list(input().strip())
    S.append(row)

# Read grid T
T = []
for _ in range(M):
    row = list(input().strip())
    T.append(row)

# Function to check if subgrid matches T
def check_match(start_i, start_j):
    for i in range(M):
        for j in range(M):
            if S[start_i + i][start_j + j] != T[i][j]:
                return False
    return True

# Find matching position
for i in range(N - M + 1):
    for j in range(N - M + 1):
        if check_match(i, j):
            # Found match, print position (1-based indexing)
            print(i + 1, j + 1)
            exit()