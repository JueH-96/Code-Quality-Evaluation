# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Dictionary to store the indices of each number
indices = defaultdict(list)

# Populate the dictionary with indices
for i in range(3*N):
    indices[A[i]].append(i)

# Sort the numbers based on the middle index
sorted_numbers = sorted(range(1, N+1), key=lambda x: indices[x][1])

# Print the sorted numbers
print(" ".join(map(str, sorted_numbers)))