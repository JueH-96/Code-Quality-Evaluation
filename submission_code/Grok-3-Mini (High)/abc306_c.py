import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Read N from the first element
index = 0
N = int(data[index])
index += 1

# Read the sequence A of length 3N
A = list(map(int, data[index:index + 3 * N]))

# Create a list to store positions for each number from 0 to N, ignoring index 0
positions = [[] for _ in range(N + 1)]

# Iterate through the sequence and store 1-based indices
for idx in range(3 * N):
    val = A[idx]
    positions[val].append(idx + 1)  # Store 1-based index

# Sort the numbers from 1 to N based on the middle occurrence index f(i)
result = sorted(range(1, N + 1), key=lambda x: positions[x][1])

# Print the result separated by spaces
print(' '.join(map(str, result)))