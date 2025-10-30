import sys

# Read input
data = sys.stdin.read().split()
N = int(data[0])
M = int(data[1])

# Calculate K
K = M - 10 * (N - 1)

# List to store all valid A sequences
sequences = []

# Backtracking function to generate non-decreasing B sequences
def backtrack(pos, low, path):
    if pos == N:
        # Create A sequence from B sequence
        A_seq = [path[i] + 10 * i for i in range(N)]
        sequences.append(A_seq)
        return
    # Iterate over possible values for B at current position
    for val in range(low, K + 1):
        path[pos] = val
        backtrack(pos + 1, val, path)

# Initialize path for B values
path = [0] * N

# Start backtracking from position 0 with lowest value 1
backtrack(0, 1, path)

# Output the number of sequences
print(len(sequences))

# Output each sequence
for seq in sequences:
    print(' '.join(map(str, seq)))