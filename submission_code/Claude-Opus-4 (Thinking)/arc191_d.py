from collections import deque

n, m, s, t = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

# BFS
initial = (s, t)
goal = (t, s)

queue = deque([(initial, 0)])
visited = {initial}

while queue:
    (a, b), d = queue.popleft()
    
    if (a, b) == goal:
        print(d)
        break
    
    # Move piece A
    for next_a in adj[a]:
        if next_a != b:  # Can't move to where B is
            state = (next_a, b)
            if state not in visited:
                visited.add(state)
                queue.append((state, d + 1))
    
    # Move piece B
    for next_b in adj[b]:
        if next_b != a:  # Can't move to where A is
            state = (a, next_b)
            if state not in visited:
                visited.add(state)
                queue.append((state, d + 1))
else:
    print(-1)