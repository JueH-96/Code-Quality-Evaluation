# YOUR CODE HERE
N, M = map(int, input().split())
X = list(map(int, input().split()))

def dist_without_edge(a, b, N, removed_edge):
    """Calculate distance from island a to island b when a specific edge is removed."""
    if a == b:
        return 0
    
    # Calculate both possible paths
    clockwise_dist = (b - a) % N
    counter_dist = (a - b) % N
    
    # Check if removed edge blocks clockwise path
    r1, r2 = removed_edge
    clockwise_blocked = False
    
    # Simulate clockwise path
    curr = a
    for _ in range(clockwise_dist):
        next_node = curr % N + 1
        if (curr == r1 and next_node == r2) or (curr == r2 and next_node == r1):
            clockwise_blocked = True
            break
        curr = next_node
    
    # Check if removed edge blocks counter-clockwise path
    counter_blocked = False
    curr = a
    for _ in range(counter_dist):
        prev_node = (curr - 2) % N + 1
        if (curr == r1 and prev_node == r2) or (curr == r2 and prev_node == r1):
            counter_blocked = True
            break
        curr = prev_node
    
    # Return shortest valid path
    if not clockwise_blocked and not counter_blocked:
        return min(clockwise_dist, counter_dist)
    elif not clockwise_blocked:
        return clockwise_dist
    else:
        return counter_dist

# Try removing each edge
min_total = float('inf')

for i in range(1, N + 1):
    # Define removed edge
    if i == N:
        removed_edge = (N, 1)
    else:
        removed_edge = (i, i + 1)
    
    # Calculate tour length with this edge removed
    total = 0
    for j in range(M - 1):
        total += dist_without_edge(X[j], X[j + 1], N, removed_edge)
    
    min_total = min(min_total, total)

print(min_total)