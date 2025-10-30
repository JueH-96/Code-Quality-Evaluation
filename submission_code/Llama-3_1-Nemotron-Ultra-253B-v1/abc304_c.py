import math
from collections import deque

n, d = map(int, input().split())
d_squared = d * d
people = [tuple(map(int, input().split())) for _ in range(n)]
infected = [False] * n
infected[0] = True
queue = deque([0])

while queue:
    u = queue.popleft()
    for v in range(n):
        if not infected[v]:
            dx = people[u][0] - people[v][0]
            dy = people[u][1] - people[v][1]
            if dx * dx + dy * dy <= d_squared:
                infected[v] = True
                queue.append(v)

for status in infected:
    print("Yes" if status else "No")