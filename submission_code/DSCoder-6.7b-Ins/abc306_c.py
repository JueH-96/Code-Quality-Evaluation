import sys
from collections import defaultdict

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Create a dictionary to store the indices of each number
indices = defaultdict(list)
for i, a in enumerate(A):
    indices[a].append(i)

# Sort the numbers by the middle index
sorted_numbers = sorted(indices.keys(), key=lambda x: indices[x][1])

# Print the sorted numbers
print(' '.join(map(str, sorted_numbers)))