from collections import deque

n, m, x1 = map(int, input().split())
trains = []
for i in range(m):
    a, b, s, t = map(int, input().split())
    trains.append((a, b, s, t))

# Build adjacency list
adj = [[] for _ in range(m)]
for i in range(m):
    for j in range(m):
        if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
            # Can transfer from train i to train j
            weight = trains[i][3] - trains[j][2]
            adj[i].append((j, weight))

# Initialize delays
x = [0] * m
x[0] = x1

# Use SPFA-like algorithm
in_queue = [False] * m
queue = deque([0])
in_queue[0] = True

while queue:
    u = queue.popleft()
    in_queue[u] = False
    
    for v, w in adj[u]:
        if x[v] < x[u] + w:
            x[v] = x[u] + w
            if not in_queue[v]:
                queue.append(v)
                in_queue[v] = True

# Output X_2, ..., X_M
print(' '.join(map(str, x[1:])))