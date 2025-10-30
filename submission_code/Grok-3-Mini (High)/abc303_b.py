import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to 0
index = 0

# Read N
N = int(data[index])
index += 1

# Read M
M = int(data[index])
index += 1

# Set to store all unique adjacent pairs
adjacent_set = set()

# Process each photo
for _ in range(M):
    # Read the permutation for this photo
    perm = [int(data[index + j]) for j in range(N)]
    # Move index forward by N
    index += N
    
    # Add adjacent pairs for this photo
    for i in range(N - 1):
        p1 = perm[i]
        p2 = perm[i + 1]
        # Store as an ordered tuple to handle unordered pairs
        pair = (min(p1, p2), max(p1, p2))
        adjacent_set.add(pair)

# Calculate total possible pairs
total_pairs = N * (N - 1) // 2

# Calculate bad mood pairs
bad_mood_pairs = total_pairs - len(adjacent_set)

# Output the result
print(bad_mood_pairs)