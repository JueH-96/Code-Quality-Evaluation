# YOUR CODE HERE
n = int(input())
a = [0] + list(map(int, input().split()))  # Make a 1-indexed

# Find a cycle
visited = {}
current = 1  # Start from vertex 1
step = 0

while current not in visited:
    visited[current] = step
    current = a[current]
    step += 1

# Now current is the first vertex we revisit, which is part of the cycle
# The cycle consists of vertices from visited[current] to step-1
cycle_start_step = visited[current]
cycle_length = step - cycle_start_step

# Reconstruct the cycle
cycle = []
for _ in range(cycle_length):
    cycle.append(current)
    current = a[current]

# Output
print(cycle_length)
print(' '.join(map(str, cycle)))