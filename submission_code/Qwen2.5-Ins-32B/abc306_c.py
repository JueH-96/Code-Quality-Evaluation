import sys

N = int(input())
A = list(map(int, input().split()))

# Dictionary to store the indices of each number
indices = {i: [] for i in range(1, N + 1)}

# Populate the dictionary with indices
for i, a in enumerate(A):
    indices[a].append(i)

# Sort the numbers based on the middle index
sorted_indices = sorted(indices.keys(), key=lambda x: indices[x][1])

# Print the sorted sequence
print(' '.join(map(str, sorted_indices)))