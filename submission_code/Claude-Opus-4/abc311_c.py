# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed for easier handling
for i in range(n):
    a[i] -= 1

# Find a cycle
visited = [-1] * n  # -1 means not visited, otherwise stores the order of visit
path = []
current = 0  # Start from vertex 0 (1 in 1-indexed)
order = 0

while visited[current] == -1:
    visited[current] = order
    path.append(current)
    current = a[current]
    order += 1

# Now current is a vertex we've already visited
# The cycle starts from current
cycle_start_index = visited[current]
cycle = path[cycle_start_index:]

# Convert back to 1-indexed and output
print(len(cycle))
print(' '.join(str(v + 1) for v in cycle))