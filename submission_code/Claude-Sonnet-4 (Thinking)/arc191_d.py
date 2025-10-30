from collections import deque

n, m, s, t = map(int, input().split())

adj = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

queue = deque([(s, t, 0)])
visited = set()
visited.add((s, t))

while queue:
    pos_a, pos_b, steps = queue.popleft()
    
    if pos_a == t and pos_b == s:
        print(steps)
        exit()
    
    for next_a in adj[pos_a]:
        if next_a != pos_b and (next_a, pos_b) not in visited:
            visited.add((next_a, pos_b))
            queue.append((next_a, pos_b, steps + 1))
    
    for next_b in adj[pos_b]:
        if next_b != pos_a and (pos_a, next_b) not in visited:
            visited.add((pos_a, next_b))
            queue.append((pos_a, next_b, steps + 1))

print(-1)