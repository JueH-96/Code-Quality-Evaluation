# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed for easier array access
# a[i] represents where vertex (i+1) points to
for i in range(n):
    a[i] -= 1

# Start from vertex 0 and follow the path
visited = {}  # vertex -> position in path
path = []
current = 0

while current not in visited:
    visited[current] = len(path)
    path.append(current)
    current = a[current]

# Now current is a vertex we've seen before
# The cycle starts from visited[current] position
cycle_start = visited[current]
cycle = path[cycle_start:]

# Convert back to 1-indexed and output
print(len(cycle))
print(*[v + 1 for v in cycle])