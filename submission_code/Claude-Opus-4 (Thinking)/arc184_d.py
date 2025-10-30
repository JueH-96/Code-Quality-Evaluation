# YOUR CODE HERE
n = int(input())
balls = []
for i in range(n):
    x, y = map(int, input().split())
    balls.append((x, y))

# Precompute which balls dominate which
dominates = [[] for _ in range(n)]
for i in range(n):
    xi, yi = balls[i]
    for j in range(n):
        if i != j:
            xj, yj = balls[j]
            if (xj < xi and yj < yi) or (xj > xi and yj > yi):
                dominates[i].append(j)

# BFS to find all reachable sets
from collections import deque
initial_state = (1 << n) - 1  # All balls present
visited = set([initial_state])
queue = deque([initial_state])

while queue:
    state = queue.popleft()
    # Try choosing each present ball
    for i in range(n):
        if state & (1 << i):  # Ball i is present
            # Remove all balls dominated by i
            new_state = state
            for j in dominates[i]:
                new_state &= ~(1 << j)  # Remove ball j
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)

print(len(visited) % 998244353)