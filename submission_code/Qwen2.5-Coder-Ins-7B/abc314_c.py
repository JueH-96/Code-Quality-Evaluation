# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
C = list(map(int, data[3:]))

# Create a list of indices for each color
color_indices = [[] for _ in range(M + 1)]
for i in range(N):
    color_indices[C[i]].append(i)

# Perform the right circular shift for each color
result = list(S)
for i in range(1, M + 1):
    indices = color_indices[i]
    k = len(indices)
    for j in range(k):
        result[indices[j]] = S[indices[(j - 1) % k]]

# Print the final result
print(''.join(result))