# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
X = list(map(int, data[2:N+2]))
A = list(map(int, data[N+2:]))

# Find the cycle in X
cycle = []
visited = [False] * (N + 1)
current = X[0]
while not visited[current]:
    visited[current] = True
    cycle.append(current)
    current = X[current - 1]

# Determine the length of the cycle
cycle_length = len(cycle)

# Determine the position in the cycle after K operations
position = (K - 1) % cycle_length

# Create the result sequence
result = A[:]
for i in range(N):
    if X[i] in cycle:
        result[i] = A[cycle[(cycle.index(X[i]) + position) % cycle_length] - 1]

# Print the result
print(" ".join(map(str, result)))