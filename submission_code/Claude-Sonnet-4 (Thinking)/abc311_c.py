n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed
adj = [0] * n
for i in range(n):
    adj[i] = a[i] - 1

# Find a cycle starting from vertex 0
visited = [False] * n
path = []
current = 0

while not visited[current]:
    visited[current] = True
    path.append(current)
    current = adj[current]

# Now current is the first repeated vertex
# Find where the cycle starts
cycle_start = path.index(current)
cycle = path[cycle_start:]

# Convert back to 1-indexed and output
print(len(cycle))
print(' '.join(str(v + 1) for v in cycle))