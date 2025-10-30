def find_subgrid(N, M, S, T):
    # Iterate over all possible starting positions in S for the top-left corner of T
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            # Check if the MxM grid T matches the subgrid in S starting at (a, b)
            match = True
            for i in range(M):
                for j in range(M):
                    if S[a + i][b + j] != T[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                # Return the 1-based indices
                return a + 1, b + 1

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# First line contains N and M
N, M = map(int, data[0].split())

# Next N lines contain the grid S
S = [data[i + 1] for i in range(N)]

# Next M lines contain the grid T
T = [data[N + i + 1] for i in range(M)]

# Find the subgrid
a, b = find_subgrid(N, M, S, T)

# Print the result
print(a, b)