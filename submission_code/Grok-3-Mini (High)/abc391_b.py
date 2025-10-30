import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Parse N and M
N = int(data[0])
M = int(data[1])

# Extract grid S and T
S = data[2:2 + N]
T = data[2 + N:2 + N + M]

# Iterate over all possible starting positions
for start_row in range(N - M + 1):
    for start_col in range(N - M + 1):
        # Check if the subgrid matches T
        if all(S[start_row + i][start_col + j] == T[i][j] for i in range(M) for j in range(M)):
            # Output the 1-based coordinates and stop
            print(start_row + 1, start_col + 1)
            sys.exit()